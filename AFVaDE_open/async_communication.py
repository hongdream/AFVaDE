from __future__ import annotations

import copy
import heapq
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from experiment import validation
from model import VaDE, gmm_sampling

LOCAL_COMPLETE = "local_complete"
AVAILABILITY_CHECK = "availability_check"
GMM_KEYS = {"rho_", "mu_c", "log_sigma2_c"}
NUMERICAL_EPS = 1e-12


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


def client_shared_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    return clone_state_dict(
        {key: value for key, value in state_dict.items() if key not in GMM_KEYS}
    )


def compute_staleness_freshness(
        staleness: int,
        a: float,
        h_min: float,
        staleness_cap: int = -1) -> float:

    if staleness < 0:
        raise ValueError(f"staleness must be non-negative, got {staleness}")
    effective_staleness = (
        min(staleness, staleness_cap)
        if staleness_cap > 0
        else staleness
    )
    freshness = (1.0 + float(effective_staleness)) ** (-float(a))
    return max(float(h_min), freshness)


def compute_shared_delta(
        end_state: Mapping[str, torch.Tensor],
        start_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:

    if set(end_state) != set(start_state):
        missing = sorted(set(start_state) - set(end_state))
        extra = sorted(set(end_state) - set(start_state))
        raise ValueError(
            "Shared state keys changed during local training: "
            f"missing={missing}, extra={extra}"
        )

    delta: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, start_tensor in start_state.items():
            end_tensor = end_state[key]
            if end_tensor.shape != start_tensor.shape:
                raise ValueError(f"Shared tensor shape changed for {key}")
            if not torch.is_floating_point(end_tensor):
                raise TypeError(
                    f"V6 delta aggregation only supports floating shared tensors: {key}"
                )
            delta[key] = (
                end_tensor.detach().cpu() - start_tensor.detach().cpu()
            ).clone()
    return delta


def compute_v6_client_weights(
        base_raw_weights: np.ndarray,
        mean_freshness: np.ndarray,
        server_beta0: float) -> tuple[np.ndarray, float, np.ndarray, float]:

    base_raw_weights = np.asarray(base_raw_weights, dtype=np.float64)
    mean_freshness = np.asarray(mean_freshness, dtype=np.float64)
    if base_raw_weights.ndim != 1 or base_raw_weights.size == 0:
        raise ValueError("base_raw_weights must be a non-empty 1-D array")
    if base_raw_weights.shape != mean_freshness.shape:
        raise ValueError("base_raw_weights and mean_freshness must have equal shape")
    base_denominator = float(base_raw_weights.sum())
    if not np.isfinite(base_denominator) or base_denominator <= NUMERICAL_EPS:
        raise ValueError("Unique-client base-weight denominator must be positive")

    base_weights = base_raw_weights / base_denominator
    batch_freshness = float(np.dot(base_weights, mean_freshness))
    if not np.isfinite(batch_freshness) or batch_freshness <= NUMERICAL_EPS:
        raise ValueError("batch_freshness must be positive")

    client_importance = base_weights * mean_freshness / batch_freshness
    beta_t = float(server_beta0) * batch_freshness
    return base_weights, batch_freshness, client_importance, beta_t


def compute_v6_gmm_weights(
        base_raw_weights: np.ndarray,
        selected_freshness: np.ndarray,
        gamma: float) -> tuple[np.ndarray, np.ndarray]:

    base_raw_weights = np.asarray(base_raw_weights, dtype=np.float64)
    selected_freshness = np.asarray(selected_freshness, dtype=np.float64)
    if base_raw_weights.shape != selected_freshness.shape:
        raise ValueError("GMM base weights and freshness must have equal shape")
    gmm_freshness = (1.0 - float(gamma)) + float(gamma) * selected_freshness
    gmm_raw_weights = base_raw_weights * gmm_freshness
    denominator = float(gmm_raw_weights.sum())
    if not np.isfinite(denominator) or denominator <= NUMERICAL_EPS:
        raise ValueError("GMM client-weight denominator must be positive")
    return gmm_freshness, gmm_raw_weights / denominator


@dataclass(frozen=True)
class AsyncAFVaDEConfig:

    seed: int
    client_count: int
    buffer_size: int = 3
    max_aggregations: int = 100
    pre_aggregations: int = 0
    slow_client_count: int = 3
    fast_delay: tuple[float, float] = (1.0, 2.0)
    slow_delay: tuple[float, float] = (5.0, 8.0)
    online_probability: float = 1.0
    availability_interval: float = 1.0
    eval_interval: int = 1
    max_processed_events: int = 1_000_000
    server_beta0: float = 1.0
    staleness_a: float = 0.5
    staleness_h_min: float = 0.1
    staleness_cap: int = -1
    gmm_staleness_gamma: float = 0.0
    buffer_duplicate_policy: str = "v6_unique"
    client_gmm_k_strategy: str = "local"

    def __post_init__(self) -> None:
        if self.client_count <= 0:
            raise ValueError("client_count must be positive")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.max_aggregations <= 0:
            raise ValueError("max_aggregations must be positive")
        if not 0 <= self.pre_aggregations < self.max_aggregations:
            raise ValueError(
                "pre_aggregations must be in [0, max_aggregations)"
            )
        if not 0 <= self.slow_client_count <= self.client_count:
            raise ValueError("slow_client_count must be in [0, client_count]")
        if not 0.0 <= self.online_probability <= 1.0:
            raise ValueError("online_probability must be in [0, 1]")
        if self.availability_interval <= 0:
            raise ValueError("availability_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.server_beta0 <= 0:
            raise ValueError("server_beta0 must be positive")
        if self.staleness_a < 0:
            raise ValueError("staleness_a must be non-negative")
        if not 0.0 <= self.staleness_h_min <= 1.0:
            raise ValueError("staleness_h_min must be in [0, 1]")
        if not 0.0 <= self.gmm_staleness_gamma <= 1.0:
            raise ValueError("gmm_staleness_gamma must be in [0, 1]")
        if self.buffer_duplicate_policy not in {"v6_unique", "separate"}:
            raise ValueError(
                "buffer_duplicate_policy must be 'v6_unique' or 'separate'"
            )
        if self.client_gmm_k_strategy not in {
                "global", "local", "nonparametric"}:
            raise ValueError(
                "client_gmm_k_strategy must be 'global', 'local', "
                "or 'nonparametric'"
            )
        self._validate_delay("fast_delay", self.fast_delay)
        self._validate_delay("slow_delay", self.slow_delay)

    @staticmethod
    def _validate_delay(name: str, interval: tuple[float, float]) -> None:
        low, high = interval
        if low <= 0 or high < low:
            raise ValueError(f"{name} must satisfy 0 < low <= high")


@dataclass(order=True)
class Event:

    time: float
    priority: int
    client_id: int
    sequence: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False, default=None)


@dataclass(frozen=True)
class PendingLocalTask:

    client_id: int
    start_version: int
    dispatch_time: float
    finish_time: float
    sampled_delay: float
    training_phase: str
    global_model_snapshot: dict[str, torch.Tensor]


@dataclass(frozen=True)
class BufferedAFVaDEUpdate:

    client_id: int
    start_version: int
    dispatch_time: float
    finish_time: float
    sampled_delay: float
    training_phase: str
    model_state: dict[str, torch.Tensor]
    shared_delta: dict[str, torch.Tensor]
    local_loss: float
    arrival_version: int
    arrival_staleness: int


@dataclass
class ClientRuntimeState:
    client_id: int
    group: str
    status: str = "idle"
    upload_count: int = 0
    local_train_executions: int = 0
    availability_checks: int = 0
    availability_failures: int = 0
    gmm_initialized: bool = False
    gmm_components: int | None = None


@dataclass(frozen=True)
class AsyncAFVaDEResult:
    config: AsyncAFVaDEConfig
    slow_client_ids: tuple[int, ...]
    events: tuple[dict[str, Any], ...]
    aggregations: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    final_server_state: dict[str, torch.Tensor]


class AFVaDEAsyncTrainer:

    def __init__(
            self,
            args: Any,
            config: AsyncAFVaDEConfig,
            server: dict[str, Any],
            clients: list[dict[str, Any]],
            data_weights: np.ndarray,
            device: torch.device,
            *,
            slow_client_ids: Iterable[int] | None = None) -> None:
        if len(clients) != config.client_count:
            raise ValueError(
                f"Config expects {config.client_count} clients, got {len(clients)}"
            )

        self.args = args
        self.config = config
        self.server = server
        self.clients = clients
        self.data_weights = np.asarray(data_weights, dtype=np.float64)
        self.data_weights = self.data_weights / self.data_weights.sum()
        self.device = device


        assignment_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 5_000])
        )
        if slow_client_ids is None:
            chosen = assignment_rng.choice(
                config.client_count,
                size=config.slow_client_count,
                replace=False,
            )
            slow_client_ids = sorted(int(value) for value in chosen)
        self.slow_client_ids = tuple(sorted(int(value) for value in slow_client_ids))
        if len(self.slow_client_ids) != config.slow_client_count:
            raise ValueError("slow_client_ids length does not match slow_client_count")

        self.client_states = {
            client_id: ClientRuntimeState(
                client_id=client_id,
                group="slow" if client_id in self.slow_client_ids else "fast",
            )
            for client_id in range(config.client_count)
        }


        self.delay_rngs = {
            client_id: np.random.default_rng(
                np.random.SeedSequence([config.seed, 2_000, client_id])
            )
            for client_id in range(config.client_count)
        }
        self.availability_rngs = {
            client_id: np.random.default_rng(
                np.random.SeedSequence([config.seed, 3_000, client_id])
            )
            for client_id in range(config.client_count)
        }
        self.gmm_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 4_000])
        )

        self.gmm_init_rng = np.random.default_rng(
            np.random.SeedSequence([config.seed, 6_000])
        )


        self.global_client_state = client_shared_state(server["vade"].state_dict())
        self.server_version = 0
        self.virtual_time = 0.0
        self.training_phase = (
            "pretrain" if config.pre_aggregations > 0 else "gmm_bootstrap"
        )
        self.server_gmm_initialized = False
        self.buffer: list[BufferedAFVaDEUpdate] = []
        self.event_queue: list[Event] = []
        self.event_rows: list[dict[str, Any]] = []
        self.aggregation_rows: list[dict[str, Any]] = []
        self.event_sequence = 0
        self.log_sequence = 0
        self.local_train_sequence = 0
        self.processed_events = 0

    def run(self) -> AsyncAFVaDEResult:

        if self.config.pre_aggregations == 0:

            self._perform_gmm_bootstrap(time=0.0)
        else:

            for client_id in range(self.config.client_count):
                self._dispatch(client_id, time=0.0, reason="initial")

        while self.server_version < self.config.max_aggregations:
            if not self.event_queue:
                raise RuntimeError("No pending event before reaching stop target")
            if self.processed_events >= self.config.max_processed_events:
                raise RuntimeError("Maximum processed-event safeguard reached")

            current_time = self.event_queue[0].time
            self.virtual_time = current_time
            same_time = self._pop_events_at(current_time)
            local_events = sorted(
                (event for event in same_time if event.kind == LOCAL_COMPLETE),
                key=lambda event: (event.client_id, event.sequence),
            )
            checks = sorted(
                (event for event in same_time if event.kind == AVAILABILITY_CHECK),
                key=lambda event: (event.client_id, event.sequence),
            )

            completed_clients: list[int] = []
            stopped = False
            for index, event in enumerate(local_events):
                should_check_availability = self._handle_local_complete(event)
                if should_check_availability:
                    completed_clients.append(event.client_id)
                self.processed_events += 1

                if self.server_version >= self.config.max_aggregations:

                    for pending_event in local_events[index + 1:] + checks:
                        heapq.heappush(self.event_queue, pending_event)
                    stopped = True
                    break

            if stopped:
                break


            for client_id in completed_clients:
                checks.append(
                    Event(
                        time=current_time,
                        priority=1,
                        client_id=client_id,
                        sequence=self._next_event_sequence(),
                        kind=AVAILABILITY_CHECK,
                    )
                )
            checks.sort(key=lambda event: (event.client_id, event.sequence))
            for event in checks:
                self._handle_availability_check(event)
                self.processed_events += 1

        return self._build_result()

    def _dispatch(self, client_id: int, *, time: float, reason: str) -> None:

        if self.training_phase not in {"pretrain", "full"}:
            raise RuntimeError(
                f"Cannot dispatch a delayed task during {self.training_phase}"
            )

        state = self.client_states[client_id]
        sampled_delay = self._sample_delay(client_id)
        finish_time = time + sampled_delay
        task = PendingLocalTask(
            client_id=client_id,
            start_version=self.server_version,
            dispatch_time=time,
            finish_time=finish_time,
            sampled_delay=sampled_delay,
            training_phase=self.training_phase,
            global_model_snapshot=clone_state_dict(self.global_client_state),
        )
        state.status = "training"
        self._push_event(finish_time, LOCAL_COMPLETE, client_id, task)
        self._log_event(
            event_time=time,
            event_type="dispatch",
            client_id=client_id,
            client_state=state.status,
            server_version=self.server_version,
            start_version=task.start_version,
            sampled_delay=sampled_delay,
            finish_time=finish_time,
            buffer_size=len(self.buffer),
            training_phase=task.training_phase,
            details=reason,
        )

    def _training_phase(self) -> str:

        return self.training_phase

    def _handle_local_complete(self, event: Event) -> bool:

        task: PendingLocalTask = event.payload
        client = self.clients[event.client_id]
        runtime = self.client_states[event.client_id]


        if task.training_phase != self.training_phase:
            self._log_event(
                event_time=self.virtual_time,
                event_type="cancelled_stale_task",
                client_id=event.client_id,
                client_state=runtime.status,
                server_version=self.server_version,
                start_version=task.start_version,
                sampled_delay=task.sampled_delay,
                finish_time=task.finish_time,
                buffer_size=len(self.buffer),
                training_phase=task.training_phase,
                details=f"current_phase={self.training_phase}",
            )
            return False


        client["vade"].load_state_dict(task.global_model_snapshot, strict=False)
        client["vade"].train()
        training_phase = task.training_phase

        if training_phase == "pretrain":
            optimizer = client["pre_optimizer"]
            lr_scheduler = client["pre_lr_s"]
            if optimizer is None or lr_scheduler is None:
                raise RuntimeError("VAE pretraining optimizer is unavailable")
        else:
            optimizer = client["optimizer"]
            lr_scheduler = client["lr_s"]

        local_loss = 0.0
        batch_count = 0
        for x, _ in client["dataloader"]:
            optimizer.zero_grad()
            x = x.to(self.device)
            if training_phase == "pretrain":


                z_mean, _ = client["vade"].encoder(x)
                reconstructed = client["vade"].decoder(z_mean)
                loss = F.mse_loss(reconstructed, x)
            else:
                loss = client["vade"].ELBO_Loss(x, self.args.alpha)
            local_loss += float(loss.detach().cpu().item())
            batch_count += 1
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
        local_loss /= max(batch_count, 1)

        execution_order = self.local_train_sequence
        self.local_train_sequence += 1
        runtime.local_train_executions += 1
        self._log_event(
            event_time=event.time,
            event_type="local_train_execute",
            client_id=event.client_id,
            client_state=runtime.status,
            server_version=self.server_version,
            start_version=task.start_version,
            sampled_delay=task.sampled_delay,
            finish_time=task.finish_time,
            buffer_size=len(self.buffer),
            training_phase=training_phase,
            execution_order=execution_order,
            local_loss=local_loss,
            details="single_machine_lazy_execution",
        )

        arrival_version = self.server_version
        arrival_staleness = arrival_version - task.start_version
        local_model_state = clone_state_dict(client["vade"].state_dict())
        local_shared_state = client_shared_state(local_model_state)
        shared_delta = compute_shared_delta(
            local_shared_state,
            task.global_model_snapshot,
        )
        packet = BufferedAFVaDEUpdate(
            client_id=event.client_id,
            start_version=task.start_version,
            dispatch_time=task.dispatch_time,
            finish_time=task.finish_time,
            sampled_delay=task.sampled_delay,
            training_phase=training_phase,
            model_state=local_model_state,
            shared_delta=shared_delta,
            local_loss=local_loss,
            arrival_version=arrival_version,
            arrival_staleness=arrival_staleness,
        )
        self.buffer.append(packet)
        runtime.status = "awaiting_activation"
        runtime.upload_count += 1
        self._log_event(
            event_time=event.time,
            event_type="upload",
            client_id=event.client_id,
            client_state=runtime.status,
            server_version=self.server_version,
            start_version=task.start_version,
            sampled_delay=task.sampled_delay,
            finish_time=task.finish_time,
            buffer_size=len(self.buffer),
            training_phase=training_phase,
            arrival_staleness=arrival_staleness,
            execution_order=execution_order,
            local_loss=local_loss,
        )

        if (
                len(self.buffer) >= self.config.buffer_size
                and self.server_version < self.config.max_aggregations):
            batch = self.buffer[:self.config.buffer_size]
            del self.buffer[:self.config.buffer_size]
            self._aggregate(batch, event.time)


        return task.training_phase == self.training_phase

    def _collect_client_latents(
            self,
            client: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:

        model = client["vade"]
        was_training = model.training
        model.eval()
        latent_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for x, y in client["dataloader"]:
                z_mean, _ = model.encoder(x.to(self.device))
                latent_batches.append(z_mean.detach().cpu())
                label_batches.append(y.detach().cpu())
        if was_training:
            model.train()
        if not latent_batches:
            raise ValueError(f"Client {client['client_id']} has no local samples")
        return (
            torch.cat(latent_batches, dim=0).numpy(),
            torch.cat(label_batches, dim=0).numpy(),
        )

    def _estimate_nonparametric_components(
            self,
            client_id: int,
            latent_z: np.ndarray) -> int:

        raise NotImplementedError(
            "client_gmm_k_strategy='nonparametric' is a reserved extension "
            "point and has not been implemented. Extend "
            "AFVaDEAsyncTrainer._estimate_nonparametric_components() first. "
            f"client_id={client_id}, sample_count={len(latent_z)}"
        )

    def _resolve_client_gmm_components(
            self,
            client_id: int,
            latent_z: np.ndarray,
            labels: np.ndarray) -> int:

        strategy = self.config.client_gmm_k_strategy
        if strategy == "global":
            component_count = int(self.args.nClusters)
        elif strategy == "local":

            component_count = int(np.unique(labels).size)
        else:
            component_count = int(
                self._estimate_nonparametric_components(client_id, latent_z)
            )

        if component_count <= 0:
            raise ValueError(
                f"Client {client_id} resolved an invalid GMM K={component_count}"
            )
        if component_count > len(latent_z):
            raise ValueError(
                f"Client {client_id} has {len(latent_z)} samples, fewer than "
                f"the requested GMM components K={component_count}"
            )
        return component_count

    def _initialize_client_gmm(
            self,
            client_id: int,
            shared_snapshot: Mapping[str, torch.Tensor],
            time: float) -> dict[str, torch.Tensor]:

        client = self.clients[client_id]
        runtime = self.client_states[client_id]
        if runtime.gmm_initialized:
            raise RuntimeError(f"Client {client_id} GMM was initialized twice")


        client["vade"].load_state_dict(shared_snapshot, strict=False)
        latent_z, labels = self._collect_client_latents(client)
        component_count = self._resolve_client_gmm_components(
            client_id,
            latent_z,
            labels,
        )
        fit_seed = int(self.gmm_init_rng.integers(0, np.iinfo(np.int32).max))
        fitted_gmm = GaussianMixture(
            n_components=component_count,
            covariance_type="diag",
            random_state=fit_seed,
            reg_covar=1e-6,
            n_init=5,
        ).fit(latent_z)

        model = client["vade"]
        rho = torch.log(torch.as_tensor(
            fitted_gmm.weights_,
            dtype=torch.float32,
            device=self.device,
        ) + NUMERICAL_EPS)
        mu = torch.as_tensor(
            fitted_gmm.means_,
            dtype=torch.float32,
            device=self.device,
        )
        log_sigma2 = torch.log(torch.as_tensor(
            fitted_gmm.covariances_,
            dtype=torch.float32,
            device=self.device,
        ).clamp_min(NUMERICAL_EPS))


        model.rho_ = torch.nn.Parameter(rho.clone())
        model.mu_c = torch.nn.Parameter(mu.clone())
        model.log_sigma2_c = torch.nn.Parameter(log_sigma2.clone())
        model.args.nClusters = component_count
        client["optimizer"] = Adam(model.parameters(), lr=self.args.lr)
        client["lr_s"] = StepLR(
            client["optimizer"],
            step_size=self.args.stepsize,
            gamma=self.args.gamma,
        )
        client["pre_optimizer"] = None
        client["pre_lr_s"] = None

        runtime.gmm_initialized = True
        runtime.gmm_components = component_count
        runtime.status = "waiting_server_gmm_init"
        self._log_event(
            event_time=time,
            event_type="client_gmm_init",
            client_id=client_id,
            client_state=runtime.status,
            training_phase="gmm_bootstrap",
            server_version=self.server_version,
            start_version=self.server_version,
            sampled_delay=0.0,
            finish_time=time,
            buffer_size=0,
            gmm_components=component_count,
            gmm_k_strategy=self.config.client_gmm_k_strategy,
            details=json.dumps(
                {
                    "sample_count": int(len(latent_z)),
                    "fit_seed": fit_seed,
                    "virtual_cost": 0.0,
                },
                separators=(",", ":"),
            ),
        )
        return {
            "rho_": rho.detach().cpu().clone(),
            "mu_c": mu.detach().cpu().clone(),
            "log_sigma2_c": log_sigma2.detach().cpu().clone(),
        }

    def _initialize_server_gmm(
            self,
            clients_gmm_param: list[dict[str, torch.Tensor]],
            time: float) -> None:

        if len(clients_gmm_param) != self.config.client_count:
            raise ValueError("Server GMM initialization requires every client once")
        sampling_seed = int(
            self.gmm_init_rng.integers(0, np.iinfo(np.int32).max)
        )
        Z_hat, _, _, _, _, _ = gmm_sampling(
            clients_gmm_param,
            torch.as_tensor(self.data_weights, dtype=torch.float32),
            self.args.sample,
            random_state=sampling_seed,
        )
        global_components = int(self.args.nClusters)
        if len(Z_hat) < global_components:
            raise ValueError(
                "GMM initialization sample count must be at least global K"
            )
        fit_seed = int(self.gmm_init_rng.integers(0, np.iinfo(np.int32).max))
        fitted_gmm = GaussianMixture(
            n_components=global_components,
            covariance_type="diag",
            random_state=fit_seed,
            reg_covar=1e-6,
            n_init=10,
        ).fit(Z_hat)

        server_model = self.server["vade"]
        with torch.no_grad():
            server_model.rho_.copy_(torch.log(torch.as_tensor(
                fitted_gmm.weights_,
                dtype=torch.float32,
                device=self.device,
            ) + NUMERICAL_EPS))
            server_model.mu_c.copy_(torch.as_tensor(
                fitted_gmm.means_,
                dtype=torch.float32,
                device=self.device,
            ))
            server_model.log_sigma2_c.copy_(torch.log(torch.as_tensor(
                fitted_gmm.covariances_,
                dtype=torch.float32,
                device=self.device,
            ).clamp_min(NUMERICAL_EPS)))

        self.server_gmm_initialized = True
        self._log_event(
            event_time=time,
            event_type="server_gmm_init",
            client_id=None,
            client_state="initialized",
            training_phase="gmm_bootstrap",
            server_version=self.server_version,
            start_version=self.server_version,
            sampled_delay=0.0,
            finish_time=time,
            buffer_size=0,
            unique_client_num=len(clients_gmm_param),
            gmm_components=global_components,
            gmm_k_strategy="global",
            details=json.dumps(
                {
                    "sampling_seed": sampling_seed,
                    "fit_seed": fit_seed,
                    "sample_count": int(len(Z_hat)),
                    "virtual_cost": 0.0,
                },
                separators=(",", ":"),
            ),
        )

    def _perform_gmm_bootstrap(self, time: float) -> None:

        if self.server_gmm_initialized:
            raise RuntimeError("Server GMM bootstrap was executed twice")
        self.training_phase = "gmm_bootstrap"


        pending_events = sorted(self.event_queue)
        self.event_queue.clear()
        for pending_event in pending_events:
            pending_task = (
                pending_event.payload
                if pending_event.kind == LOCAL_COMPLETE
                else None
            )
            self._log_event(
                event_time=time,
                event_type=(
                    "cancelled_warmup_task"
                    if pending_event.kind == LOCAL_COMPLETE
                    else "cancelled_availability_check"
                ),
                client_id=pending_event.client_id,
                client_state=self.client_states[pending_event.client_id].status,
                training_phase=(
                    pending_task.training_phase
                    if pending_task is not None
                    else "pretrain"
                ),
                server_version=self.server_version,
                start_version=(
                    pending_task.start_version
                    if pending_task is not None
                    else None
                ),
                sampled_delay=(
                    pending_task.sampled_delay
                    if pending_task is not None
                    else None
                ),
                finish_time=pending_event.time,
                buffer_size=len(self.buffer),
                details="cancelled_at_zero_time_phase_barrier",
            )

        discarded_updates = len(self.buffer)
        self.buffer.clear()
        self._log_event(
            event_time=time,
            event_type="gmm_bootstrap_start",
            client_id=None,
            client_state=None,
            training_phase="gmm_bootstrap",
            server_version=self.server_version,
            start_version=self.server_version,
            sampled_delay=0.0,
            finish_time=time,
            buffer_size=0,
            details=json.dumps(
                {
                    "discarded_warmup_buffer_updates": discarded_updates,
                    "cancelled_pending_events": len(pending_events),
                    "virtual_cost": 0.0,
                },
                separators=(",", ":"),
            ),
        )

        shared_snapshot = clone_state_dict(self.global_client_state)
        clients_gmm_param = [
            self._initialize_client_gmm(client_id, shared_snapshot, time)
            for client_id in range(self.config.client_count)
        ]
        self._initialize_server_gmm(clients_gmm_param, time)

        self.training_phase = "full"
        self._log_event(
            event_time=time,
            event_type="gmm_bootstrap_complete",
            client_id=None,
            client_state=None,
            training_phase="gmm_bootstrap",
            server_version=self.server_version,
            start_version=self.server_version,
            sampled_delay=0.0,
            finish_time=time,
            buffer_size=0,
            unique_client_num=self.config.client_count,
            details="formal_async_training_starts_at_same_virtual_time",
        )
        for client_id in range(self.config.client_count):
            self._dispatch(client_id, time=time, reason="post_gmm_bootstrap")

    def _build_aggregation_groups(
            self,
            batch: list[BufferedAFVaDEUpdate]) -> list[tuple[int, tuple[int, ...]]]:

        if self.config.buffer_duplicate_policy == "separate":
            return [
                (packet.client_id, (index,))
                for index, packet in enumerate(batch)
            ]

        grouped_indices: dict[int, list[int]] = {}
        for index, packet in enumerate(batch):
            grouped_indices.setdefault(packet.client_id, []).append(index)
        return [
            (client_id, tuple(indices))
            for client_id, indices in grouped_indices.items()
        ]

    def _fuse_group_deltas(
            self,
            batch: list[BufferedAFVaDEUpdate],
            groups: list[tuple[int, tuple[int, ...]]],
            freshness: np.ndarray) -> list[dict[str, torch.Tensor]]:

        expected_keys = set(self.global_client_state)
        fused_deltas: list[dict[str, torch.Tensor]] = []
        with torch.no_grad():
            for _, indices in groups:
                denominator = float(np.sum(freshness[list(indices)]))
                if denominator <= NUMERICAL_EPS:
                    raise ValueError("Client delta freshness denominator must be positive")
                fused: dict[str, torch.Tensor] = {}
                for key, current_tensor in self.global_client_state.items():
                    accumulator = torch.zeros_like(current_tensor)
                    for index in indices:
                        packet_delta = batch[index].shared_delta
                        if set(packet_delta) != expected_keys:
                            raise ValueError("Buffered shared-delta keys do not match global state")
                        delta_tensor = packet_delta[key]
                        if delta_tensor.shape != current_tensor.shape:
                            raise ValueError(f"Buffered delta shape mismatch for {key}")
                        accumulator.add_(
                            delta_tensor.to(dtype=current_tensor.dtype),
                            alpha=float(freshness[index]),
                        )
                    fused[key] = (accumulator / denominator).clone()
                fused_deltas.append(fused)
        return fused_deltas

    @staticmethod
    def _select_freshest_update(
            batch: list[BufferedAFVaDEUpdate],
            indices: tuple[int, ...]) -> int:

        return max(indices, key=lambda index: (batch[index].start_version, index))

    def _aggregate(self, batch: list[BufferedAFVaDEUpdate], time: float) -> None:

        if not batch:
            raise ValueError("Cannot aggregate an empty buffer batch")

        version_before = self.server_version
        training_phase = self.training_phase
        if training_phase not in {"pretrain", "full"}:
            raise RuntimeError(f"Cannot aggregate during {training_phase}")
        packet_phases = {packet.training_phase for packet in batch}
        if packet_phases != {training_phase}:
            raise RuntimeError(
                "A buffer batch crossed the pretrain/full boundary: "
                f"expected={training_phase}, packets={sorted(packet_phases)}"
            )
        staleness = np.asarray(
            [version_before - packet.start_version for packet in batch],
            dtype=np.int64,
        )
        freshness = np.asarray(
            [
                compute_staleness_freshness(
                    int(value),
                    self.config.staleness_a,
                    self.config.staleness_h_min,
                    self.config.staleness_cap,
                )
                for value in staleness
            ],
            dtype=np.float64,
        )


        upload_counts = np.asarray(
            [self.client_states[i].upload_count for i in range(self.config.client_count)],
            dtype=np.float64,
        )
        total_uploads = float(upload_counts.sum())
        if total_uploads > 0:
            commu_weight = self.args.xi / (
                self.args.xi + upload_counts / total_uploads
            )
        else:
            commu_weight = np.ones(self.config.client_count, dtype=np.float64)

        groups = self._build_aggregation_groups(batch)
        group_client_ids = [client_id for client_id, _ in groups]
        base_raw_weights = np.asarray(
            [
                self.data_weights[client_id] * commu_weight[client_id]
                for client_id in group_client_ids
            ],
            dtype=np.float64,
        )
        mean_client_freshness = np.asarray(
            [float(np.mean(freshness[list(indices)])) for _, indices in groups],
            dtype=np.float64,
        )
        (
            base_weights,
            batch_freshness,
            client_importance,
            beta_t,
        ) = compute_v6_client_weights(
            base_raw_weights,
            mean_client_freshness,
            self.config.server_beta0,
        )


        fused_deltas = self._fuse_group_deltas(batch, groups, freshness)
        aggregated_delta: dict[str, torch.Tensor] = {}
        updated_shared_state: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for key, current_tensor in self.global_client_state.items():
                delta_sum = torch.zeros_like(current_tensor)
                for importance, fused_delta in zip(client_importance, fused_deltas):
                    delta_sum.add_(
                        fused_delta[key].to(dtype=current_tensor.dtype),
                        alpha=float(importance),
                    )
                aggregated_delta[key] = delta_sum.clone()

                updated_shared_state[key] = (
                    current_tensor.detach().cpu()
                    + float(beta_t) * aggregated_delta[key]
                ).clone()
        self.server["vade"].load_state_dict(updated_shared_state, strict=False)


        selected_gmm_indices = [
            self._select_freshest_update(batch, indices)
            for _, indices in groups
        ]
        selected_gmm_freshness = freshness[selected_gmm_indices]
        server_loss_value: float | None = None
        if training_phase == "pretrain":


            gmm_freshness = np.ones(len(groups), dtype=np.float64)
            gmm_client_weights = np.zeros(len(groups), dtype=np.float64)
        else:


            gmm_freshness, gmm_client_weights = compute_v6_gmm_weights(
                base_raw_weights,
                selected_gmm_freshness,
                self.config.gmm_staleness_gamma,
            )
            gmm_packets = [batch[index] for index in selected_gmm_indices]
            clients_gmm_param = [
                {
                    "rho_": packet.model_state["rho_"],
                    "mu_c": packet.model_state["mu_c"],
                    "log_sigma2_c": packet.model_state["log_sigma2_c"],
                }
                for packet in gmm_packets
            ]
            gmm_participate_weight = torch.as_tensor(
                gmm_client_weights,
                dtype=torch.float32,
            )
            sampling_seed = int(
                self.gmm_rng.integers(0, np.iinfo(np.int32).max)
            )
            Z_hat, y_hat, lengths, _, _, _ = gmm_sampling(
                clients_gmm_param,
                gmm_participate_weight,
                self.args.sample,
                random_state=sampling_seed,
            )
            Z_hat = torch.from_numpy(Z_hat).float().to(self.device)

            self.server["vade"].train()
            X_hat = self.server["vade"].decoder(Z_hat)

            decoded_X_hat = torch.zeros_like(X_hat)
            uploaded_decoder = copy.deepcopy(self.server["vade"].decoder).to(
                self.device
            )
            start_index = 0
            with torch.no_grad():
                for packet, num_clusters_for_client in zip(gmm_packets, lengths):
                    decoder_state = {
                        key[len("decoder."):]: value
                        for key, value in packet.model_state.items()
                        if key.startswith("decoder.")
                    }
                    uploaded_decoder.load_state_dict(decoder_state)
                    for cluster_id in range(num_clusters_for_client):
                        label = start_index + cluster_id
                        data_points = np.where(y_hat == label)[0]
                        if len(data_points) == 0:
                            continue
                        tensor_index = torch.as_tensor(
                            data_points,
                            dtype=torch.long,
                            device=self.device,
                        )
                        decoded_X_hat[tensor_index] = uploaded_decoder(
                            Z_hat[tensor_index]
                        )
                    start_index += num_clusters_for_client

            server_loss = self.server["vade"].ELBO_Loss_server(
                X_hat,
                decoded_X_hat,
                self.args.alpha,
            )
            self.server["optimizer"].zero_grad()
            server_loss.backward()
            self.server["optimizer"].step()
            self.server["lr_s"].step()
            server_loss_value = float(server_loss.detach().cpu().item())


        self.global_client_state = clone_state_dict(updated_shared_state)
        self.server_version += 1
        start_gmm_bootstrap = (
            training_phase == "pretrain"
            and self.server_version == self.config.pre_aggregations
        )
        if start_gmm_bootstrap:

            for client in self.clients:
                client["pre_optimizer"] = None
                client["pre_lr_s"] = None

        metrics: dict[str, float | None]
        if self.server_version % self.config.eval_interval == 0:
            metrics = self._evaluate()
        else:
            metrics = {
                "purity": None,
                "ari": None,
                "nmi": None,
                "acc": None,
                "fm": None,
            }

        groups_are_slow = np.asarray(
            [client_id in self.slow_client_ids for client_id in group_client_ids],
            dtype=bool,
        )
        total_slow_q_vae = float(client_importance[groups_are_slow].sum())
        total_fast_q_vae = float(client_importance[~groups_are_slow].sum())
        total_slow_q_gmm = float(gmm_client_weights[groups_are_slow].sum())
        total_fast_q_gmm = float(gmm_client_weights[~groups_are_slow].sum())
        slow_effective_weight = float(beta_t * total_slow_q_vae)
        fast_effective_weight = float(beta_t * total_fast_q_vae)

        client_statistics = []
        for unit_index, ((client_id, indices), selected_index) in enumerate(
                zip(groups, selected_gmm_indices)):
            unit_staleness = staleness[list(indices)]
            client_statistics.append({
                "client_id": client_id,
                "client_group": self.client_states[client_id].group,
                "num_updates_in_buffer": len(indices),
                "data_weight": float(self.data_weights[client_id]),
                "comm_weight": float(commu_weight[client_id]),
                "base_weight": float(base_weights[unit_index]),
                "mean_client_freshness": float(mean_client_freshness[unit_index]),
                "client_importance": float(client_importance[unit_index]),
                "selected_gmm_weight": float(gmm_client_weights[unit_index]),
                "selected_update_freshness": float(
                    selected_gmm_freshness[unit_index]
                ),
                "gmm_freshness_multiplier": float(gmm_freshness[unit_index]),
                "selected_gmm_start_version": batch[selected_index].start_version,
                "min_staleness": int(np.min(unit_staleness)),
                "max_staleness": int(np.max(unit_staleness)),
            })

        row: dict[str, Any] = {
            "seed": self.config.seed,
            "aggregation_step": self.server_version,
            "virtual_time": time,
            "server_version_before": version_before,
            "server_version_after": self.server_version,
            "training_phase": training_phase,
            "buffer_size": len(batch),
            "buffer_client_ids": ";".join(str(packet.client_id) for packet in batch),
            "unique_clients": len({packet.client_id for packet in batch}),
            "unique_client_num": len({packet.client_id for packet in batch}),
            "aggregation_unit_num": len(groups),
            "buffer_duplicate_policy": self.config.buffer_duplicate_policy,
            "normalized_weights": ";".join(
                f"{value:.12g}" for value in client_importance
            ),
            "gmm_client_weights": ";".join(
                f"{value:.12g}" for value in gmm_client_weights
            ),
            "mean_staleness": float(np.mean(staleness)),
            "p95_staleness": float(np.percentile(staleness, 95)),
            "max_staleness": int(np.max(staleness)),
            "min_staleness": int(np.min(staleness)),
            "staleness_values": ";".join(str(int(value)) for value in staleness),
            "mean_freshness": float(np.mean(freshness)),
            "min_freshness": float(np.min(freshness)),
            "max_freshness": float(np.max(freshness)),
            "batch_freshness": batch_freshness,
            "beta_t": beta_t,
            "total_fast_q_vae": total_fast_q_vae,
            "total_slow_q_vae": total_slow_q_vae,
            "fast_effective_weight": fast_effective_weight,
            "slow_effective_weight": slow_effective_weight,
            "total_fast_q_gmm": total_fast_q_gmm,
            "total_slow_q_gmm": total_slow_q_gmm,
            "mean_local_loss": float(np.mean([packet.local_loss for packet in batch])),
            "server_loss": server_loss_value,
            "client_statistics": client_statistics,
            **metrics,
        }
        self.aggregation_rows.append(row)
        self._log_event(
            event_time=time,
            event_type="aggregation",
            client_id=None,
            client_state=None,
            server_version=self.server_version,
            start_version=None,
            training_phase=training_phase,
            buffer_size=len(batch),
            unique_client_num=len({packet.client_id for packet in batch}),
            aggregation_step=self.server_version,
            aggregation_staleness=float(np.mean(staleness)),
            mean_freshness=float(np.mean(freshness)),
            min_freshness=float(np.min(freshness)),
            max_freshness=float(np.max(freshness)),
            batch_freshness=batch_freshness,
            beta_t=beta_t,
            total_fast_q_vae=total_fast_q_vae,
            total_slow_q_vae=total_slow_q_vae,
            fast_effective_weight=fast_effective_weight,
            slow_effective_weight=slow_effective_weight,
            total_fast_q_gmm=total_fast_q_gmm,
            total_slow_q_gmm=total_slow_q_gmm,
            details=json.dumps(
                {
                    "buffer_duplicate_policy": self.config.buffer_duplicate_policy,
                    "training_phase": training_phase,
                    "clients": [packet.client_id for packet in batch],
                    "staleness": staleness.tolist(),
                    "freshness": freshness.tolist(),
                    "client_statistics": client_statistics,
                },
                separators=(",", ":"),
            ),
        )
        if start_gmm_bootstrap:

            self._perform_gmm_bootstrap(time)

    def _handle_availability_check(self, event: Event) -> None:
        runtime = self.client_states[event.client_id]
        if runtime.status == "training":


            self._log_event(
                event_time=event.time,
                event_type="cancelled_availability_check",
                client_id=event.client_id,
                client_state=runtime.status,
                training_phase=self.training_phase,
                server_version=self.server_version,
                start_version=None,
                buffer_size=len(self.buffer),
                details="client_already_has_dispatched_task",
            )
            return
        runtime.availability_checks += 1
        draw = float(self.availability_rngs[event.client_id].random())
        online = draw < self.config.online_probability
        runtime.status = "online" if online else "offline"
        if not online:
            runtime.availability_failures += 1

        self._log_event(
            event_time=event.time,
            event_type="availability_check",
            client_id=event.client_id,
            client_state=runtime.status,
            training_phase=self.training_phase,
            server_version=self.server_version,
            start_version=None,
            online_draw=draw,
            online=online,
            buffer_size=len(self.buffer),
        )
        if online:
            self._dispatch(event.client_id, time=event.time, reason="reactivated")
        else:
            self._push_event(
                event.time + self.config.availability_interval,
                AVAILABILITY_CHECK,
                event.client_id,
            )

    def _evaluate(self) -> dict[str, float]:

        was_training = self.server["vade"].training
        self.server["vade"].eval()
        predicted = []
        true = []
        with torch.no_grad():
            for client in self.clients:
                for x, y in client["dataloader"]:
                    x = x.to(self.device)
                    true.append(y.numpy())
                    predicted.append(self.server["vade"].predict(x))
        if was_training:
            self.server["vade"].train()

        y_true = np.concatenate(true, axis=0)
        y_pred = np.concatenate(predicted, axis=0)
        purity, ari, nmi, acc, fm = validation(y_true, y_pred)
        return {
            "purity": float(purity),
            "ari": float(ari),
            "nmi": float(nmi),
            "acc": float(acc),
            "fm": float(fm),
        }

    def _sample_delay(self, client_id: int) -> float:
        runtime = self.client_states[client_id]
        low, high = (
            self.config.slow_delay
            if runtime.group == "slow"
            else self.config.fast_delay
        )
        if low == high:
            return float(low)
        return float(self.delay_rngs[client_id].uniform(low, high))

    def _push_event(
            self,
            time: float,
            kind: str,
            client_id: int,
            payload: Any = None) -> None:
        priority = 0 if kind == LOCAL_COMPLETE else 1
        heapq.heappush(
            self.event_queue,
            Event(
                time=float(time),
                priority=priority,
                client_id=client_id,
                sequence=self._next_event_sequence(),
                kind=kind,
                payload=payload,
            ),
        )

    def _pop_events_at(self, time: float) -> list[Event]:
        events: list[Event] = []
        while self.event_queue and self.event_queue[0].time == time:
            events.append(heapq.heappop(self.event_queue))
        return events

    def _next_event_sequence(self) -> int:
        value = self.event_sequence
        self.event_sequence += 1
        return value

    def _log_event(self, **values: Any) -> None:
        row = {
            "sequence": self.log_sequence,
            "seed": self.config.seed,
            "event_time": None,
            "event_type": None,
            "client_id": None,
            "client_group": None,
            "client_state": None,
            "training_phase": None,
            "server_version": None,
            "start_version": None,
            "sampled_delay": None,
            "finish_time": None,
            "online_probability": self.config.online_probability,
            "online_draw": None,
            "online": None,
            "buffer_size": None,
            "arrival_staleness": None,
            "aggregation_staleness": None,
            "unique_client_num": None,
            "mean_freshness": None,
            "min_freshness": None,
            "max_freshness": None,
            "batch_freshness": None,
            "beta_t": None,
            "total_fast_q_vae": None,
            "total_slow_q_vae": None,
            "fast_effective_weight": None,
            "slow_effective_weight": None,
            "total_fast_q_gmm": None,
            "total_slow_q_gmm": None,
            "aggregation_step": None,
            "execution_order": None,
            "local_loss": None,
            "gmm_components": None,
            "gmm_k_strategy": None,
            "details": None,
        }
        row.update(values)
        client_id = row["client_id"]
        if client_id is not None:
            row["client_group"] = self.client_states[int(client_id)].group
        self.event_rows.append(row)
        self.log_sequence += 1

    def _build_result(self) -> AsyncAFVaDEResult:
        all_staleness = [
            int(value)
            for row in self.aggregation_rows
            for value in str(row["staleness_values"]).split(";")
        ]
        upload_counts = {
            client_id: runtime.upload_count
            for client_id, runtime in self.client_states.items()
        }
        local_train_executions = int(sum(
            runtime.local_train_executions
            for runtime in self.client_states.values()
        ))


        evaluation_repeats = max(int(getattr(self.args, "iters", 1)), 1)
        repeated_metrics = [self._evaluate() for _ in range(evaluation_repeats)]
        final_metrics = {
            name: float(np.mean([metrics[name] for metrics in repeated_metrics]))
            for name in ("purity", "ari", "nmi", "acc", "fm")
        }
        final_metrics_std = {
            name: float(np.std([metrics[name] for metrics in repeated_metrics]))
            for name in ("purity", "ari", "nmi", "acc", "fm")
        }
        summary: dict[str, Any] = {
            "seed": self.config.seed,
            "fast_delay_min": self.config.fast_delay[0],
            "fast_delay_max": self.config.fast_delay[1],
            "slow_delay_min": self.config.slow_delay[0],
            "slow_delay_max": self.config.slow_delay[1],
            "online_probability": self.config.online_probability,
            "server_beta0": self.config.server_beta0,
            "staleness_a": self.config.staleness_a,
            "staleness_h_min": self.config.staleness_h_min,
            "staleness_cap": self.config.staleness_cap,
            "gmm_staleness_gamma": self.config.gmm_staleness_gamma,
            "buffer_duplicate_policy": self.config.buffer_duplicate_policy,
            "client_gmm_k_strategy": self.config.client_gmm_k_strategy,
            "server_gmm_initialized": self.server_gmm_initialized,
            "client_gmm_components": json.dumps(
                {
                    client_id: runtime.gmm_components
                    for client_id, runtime in self.client_states.items()
                },
                separators=(",", ":"),
            ),
            "pre_aggregations": self.config.pre_aggregations,
            "full_aggregations": (
                self.config.max_aggregations - self.config.pre_aggregations
            ),
            "aggregations": self.server_version,
            "final_virtual_time": self.virtual_time,
            "total_uploads": int(sum(upload_counts.values())),
            "local_train_executions": local_train_executions,
            "availability_checks": int(sum(
                runtime.availability_checks
                for runtime in self.client_states.values()
            )),
            "availability_failures": int(sum(
                runtime.availability_failures
                for runtime in self.client_states.values()
            )),
            "pending_buffer_updates": len(self.buffer),
            "pending_events": len(self.event_queue),
            "pending_local_tasks": int(sum(
                event.kind == LOCAL_COMPLETE for event in self.event_queue
            )),
            "pending_availability_checks": int(sum(
                event.kind == AVAILABILITY_CHECK for event in self.event_queue
            )),
            "mean_client_uploads": float(np.mean(list(upload_counts.values()))),
            "std_client_uploads": float(np.std(list(upload_counts.values()))),
            "mean_staleness": float(np.mean(all_staleness)),
            "p95_staleness": float(np.percentile(all_staleness, 95)),
            "max_staleness": int(np.max(all_staleness)),
            "mean_batch_freshness": float(np.mean([
                row["batch_freshness"] for row in self.aggregation_rows
            ])),
            "mean_beta_t": float(np.mean([
                row["beta_t"] for row in self.aggregation_rows
            ])),
            "slow_client_ids": ";".join(str(value) for value in self.slow_client_ids),
            "client_upload_counts": json.dumps(upload_counts, separators=(",", ":")),
            **{f"final_{name}": value for name, value in final_metrics.items()},
            **{
                f"final_{name}_std": value
                for name, value in final_metrics_std.items()
            },
        }
        return AsyncAFVaDEResult(
            config=self.config,
            slow_client_ids=self.slow_client_ids,
            events=tuple(self.event_rows),
            aggregations=tuple(self.aggregation_rows),
            summary=summary,
            final_server_state=clone_state_dict(self.server["vade"].state_dict()),
        )

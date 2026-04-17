from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pdewm.solvers.base import BasePDESolver, SimulationResult
from pdewm.solvers.contexts import PDEContext


@dataclass(slots=True)
class KdV1DSolver(BasePDESolver):
    grid_size: int = 256
    domain_length: float = 2.0 * np.pi
    dealias: bool = True
    advective_cfl: float = 0.4
    dispersive_cfl: float = 0.4
    max_substeps_per_step: int = 4096
    solver_name: str = "kdv_1d"

    def sample_initial_condition(self, rng: np.random.Generator, context: PDEContext) -> np.ndarray:
        amplitude = float(context.parameters.get("ic_amplitude", 1.0))
        bandwidth = int(context.parameters.get("ic_bandwidth", 6))
        domain_length = float(context.parameters.get("domain_length", self.domain_length))
        x = np.linspace(0.0, domain_length, self.grid_size, endpoint=False)
        state = np.zeros_like(x)

        for mode in range(1, bandwidth + 1):
            coefficient = rng.normal(scale=amplitude / mode)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            state += coefficient * np.cos(2.0 * np.pi * mode * x / domain_length + phase)

        return state[np.newaxis, :].astype(np.float32)

    def simulate(
        self,
        initial_state: np.ndarray,
        context: PDEContext,
        num_steps: int,
        dt: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> SimulationResult:
        del options
        dt = float(dt or context.dt)
        domain_length = float(context.parameters.get("domain_length", self.domain_length))
        dispersion = float(context.parameters.get("dispersion", 1.0))
        warnings: list[str] = []

        u0 = np.asarray(initial_state, dtype=np.float64).reshape(self.grid_size)
        x = np.linspace(0.0, domain_length, self.grid_size, endpoint=False)
        dx = domain_length / self.grid_size
        wave_numbers = 2.0 * np.pi * np.fft.fftfreq(self.grid_size, d=dx)
        ik = 1j * wave_numbers
        ik3 = (1j * wave_numbers) ** 3
        forcing_hat = np.fft.fft(self._forcing(x, context.forcing_descriptor, domain_length))
        dealias_mask = self._dealias_mask(wave_numbers) if self.dealias else np.ones_like(wave_numbers)

        trajectory = np.empty((num_steps + 1, 1, self.grid_size), dtype=np.float32)
        trajectory[0, 0] = u0.astype(np.float32)
        u_hat = np.fft.fft(u0)
        start = time.perf_counter()
        status = "ok"
        total_substeps = 0
        max_substeps_used = 0

        for step in range(1, num_steps + 1):
            remaining = dt
            used_substeps = 0
            while remaining > 0.0:
                state_for_dt = np.fft.ifft(u_hat).real
                sub_dt = min(remaining, self._stable_substep(state_for_dt, dx, dispersion, wave_numbers))
                if not np.isfinite(sub_dt) or sub_dt <= 0.0:
                    warnings.append(f"invalid sub-step encountered at step {step}")
                    status = "unstable"
                    trajectory = trajectory[:step]
                    break

                u_hat = self._rk4_step(u_hat, sub_dt, dispersion, ik, ik3, forcing_hat, dealias_mask)
                remaining -= sub_dt
                used_substeps += 1
                total_substeps += 1

                if used_substeps > self.max_substeps_per_step:
                    warnings.append(
                        f"max_substeps_per_step exceeded at step {step} (>{self.max_substeps_per_step})"
                    )
                    status = "unstable"
                    trajectory = trajectory[:step]
                    break

            max_substeps_used = max(max_substeps_used, used_substeps)
            if status != "ok":
                break

            state = np.fft.ifft(u_hat).real
            if not np.isfinite(state).all():
                warnings.append(f"non-finite state detected at step {step}")
                status = "nan"
                trajectory = trajectory[:step]
                break

            if float(np.max(np.abs(state))) > 1.0e5:
                warnings.append(f"instability threshold reached at step {step}")
                status = "unstable"
                trajectory = trajectory[:step]
                break

            trajectory[step, 0] = state.astype(np.float32)

        runtime_sec = time.perf_counter() - start
        diagnostics = {
            "max_abs_value": float(np.max(np.abs(trajectory))),
            "mean_energy": float(np.mean(trajectory**2)),
            "steps_completed": int(trajectory.shape[0] - 1),
            "total_substeps": int(total_substeps),
            "max_substeps_used": int(max_substeps_used),
            "dispersion": float(dispersion),
        }
        return SimulationResult(
            trajectory=trajectory,
            status=status,
            runtime_sec=runtime_sec,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _forcing(self, x: np.ndarray, descriptor: dict[str, Any], domain_length: float) -> np.ndarray:
        amplitude = float(descriptor.get("amplitude", 0.0))
        if amplitude == 0.0:
            return np.zeros_like(x)
        mode = int(descriptor.get("mode", 1))
        phase = float(descriptor.get("phase", 0.0))
        return amplitude * np.sin(2.0 * np.pi * mode * x / domain_length + phase)

    def _dealias_mask(self, wave_numbers: np.ndarray) -> np.ndarray:
        cutoff = (2.0 / 3.0) * np.max(np.abs(wave_numbers))
        return (np.abs(wave_numbers) <= cutoff).astype(np.float64)

    def _rhs(
        self,
        u_hat: np.ndarray,
        dispersion: float,
        ik: np.ndarray,
        ik3: np.ndarray,
        forcing_hat: np.ndarray,
        dealias_mask: np.ndarray,
    ) -> np.ndarray:
        u = np.fft.ifft(u_hat).real
        nonlinear_hat = -3.0 * ik * np.fft.fft(u**2)
        nonlinear_hat = nonlinear_hat * dealias_mask
        linear_hat = -dispersion * ik3 * u_hat
        return linear_hat + nonlinear_hat + forcing_hat

    def _rk4_step(
        self,
        u_hat: np.ndarray,
        dt: float,
        dispersion: float,
        ik: np.ndarray,
        ik3: np.ndarray,
        forcing_hat: np.ndarray,
        dealias_mask: np.ndarray,
    ) -> np.ndarray:
        k1 = self._rhs(u_hat, dispersion, ik, ik3, forcing_hat, dealias_mask)
        k2 = self._rhs(u_hat + 0.5 * dt * k1, dispersion, ik, ik3, forcing_hat, dealias_mask)
        k3 = self._rhs(u_hat + 0.5 * dt * k2, dispersion, ik, ik3, forcing_hat, dealias_mask)
        k4 = self._rhs(u_hat + dt * k3, dispersion, ik, ik3, forcing_hat, dealias_mask)
        return u_hat + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _stable_substep(
        self,
        state: np.ndarray,
        dx: float,
        dispersion: float,
        wave_numbers: np.ndarray,
    ) -> float:
        speed = float(np.max(np.abs(state)))
        dt_adv = self.advective_cfl * dx / (6.0 * speed + 1.0e-8)
        max_k = float(np.max(np.abs(wave_numbers)))
        dt_disp = self.dispersive_cfl / (abs(dispersion) * (max_k**3) + 1.0e-8)
        return float(min(dt_adv, dt_disp))

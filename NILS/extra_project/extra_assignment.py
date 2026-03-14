#run with --steps [number] to increase/decrease walk length, --seed [number] to change random seed, and --step-std [number] to adjust step size variability.
"""Compare JAX and PyTorch RNG behavior via random-walk reproducibility.

This script simulates 2D random walks for two particles (A, B):

1) Batch run: both particles simulated together.
2) Serial run: A simulated alone, then B simulated alone.

Expected outcome:
- JAX (with explicit key handling): batch and serial trajectories match exactly.
- PyTorch (global RNG state): serial trajectories diverge from batch trajectories.

Run:
	python extra_project/extra_assignmet.py --steps 300 --seed 7 --step-std 0.2 --out jax_vs_torch_random_walks.png
"""

import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import torch

PARTICLE_NAMES = ("Particle A", "Particle B")
PARTICLE_COLORS = ("tab:blue", "tab:orange")

@dataclass
class WalkResult:
	batch: np.ndarray
	serial: np.ndarray


def _paths_from_increments(increments: np.ndarray) -> np.ndarray:
	"""Convert increments of shape [steps, particles, 2] into positions.

	Returns positions with shape [steps + 1, particles, 2], starting at (0, 0).
	"""
	cumulative = np.cumsum(increments, axis=0)
	start = np.zeros((1, cumulative.shape[1], cumulative.shape[2]), dtype=cumulative.dtype)
	return np.concatenate([start, cumulative], axis=0)


def simulate_jax(steps: int, seed: int, step_std: float) -> WalkResult:
	if jax is None or jnp is None:
		raise RuntimeError("JAX is not available in the active environment.")

	key = jax.random.PRNGKey(seed)
	step_keys = jax.random.split(key, steps)

	particle_ids = jnp.arange(2, dtype=jnp.uint32)

	def step_for_all_particles(step_key):
		def one_particle(pid):
			# fold_in makes each particle stream explicit and stable across batching changes.
			return jax.random.normal(jax.random.fold_in(step_key, pid), shape=(2,)) * step_std

		return jax.vmap(one_particle)(particle_ids)

	increments_batch = jax.vmap(step_for_all_particles)(step_keys)
	paths_batch = _paths_from_increments(np.array(increments_batch))

	serial_paths = []
	for pid in range(2):
		# Reconstruct each particle alone using the same per-step keys.
		increments = []
		for step_key in step_keys:
			noise = jax.random.normal(jax.random.fold_in(step_key, pid), shape=(2,)) * step_std
			increments.append(noise)
		increments = jnp.stack(increments, axis=0)[:, None, :]
		serial_paths.append(_paths_from_increments(np.array(increments))[:, 0, :])

	paths_serial = np.stack(serial_paths, axis=1)
	return WalkResult(batch=paths_batch, serial=paths_serial)


def simulate_torch(steps: int, seed: int, step_std: float) -> WalkResult:
	if torch is None:
		raise RuntimeError("PyTorch is not available in the active environment.")

	torch.manual_seed(seed)
	# In PyTorch this consumes one shared RNG stream for both particles at once.
	increments_batch = torch.randn(steps, 2, 2) * step_std
	paths_batch = _paths_from_increments(increments_batch.numpy())

	torch.manual_seed(seed)
	# Serial draws advance the same global stream in a different order than batch.
	increments_a = torch.randn(steps, 1, 2) * step_std
	increments_b = torch.randn(steps, 1, 2) * step_std
	increments_serial = torch.cat([increments_a, increments_b], dim=1)
	paths_serial = _paths_from_increments(increments_serial.numpy())

	return WalkResult(batch=paths_batch, serial=paths_serial)


def max_abs_diff(batch: np.ndarray, serial: np.ndarray) -> tuple[float, float]:
	diff = np.abs(batch - serial)
	per_particle = diff.max(axis=(0, 2))
	return float(per_particle[0]), float(per_particle[1])


def plot_comparison(jax_result: WalkResult, torch_result: WalkResult, out_path: str) -> None:
	fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

	for row, (framework_name, result) in enumerate((("JAX", jax_result), ("PyTorch", torch_result))):
		for col, (label, color) in enumerate(zip(PARTICLE_NAMES, PARTICLE_COLORS)):
			ax = axes[row, col]
			batch_xy = result.batch[:, col, :]
			serial_xy = result.serial[:, col, :]

			max_diff = float(np.max(np.abs(batch_xy - serial_xy)))
			is_match = np.allclose(batch_xy, serial_xy, rtol=0.0, atol=1e-12)

			ax.plot(
				batch_xy[:, 0],
				batch_xy[:, 1],
				color=color,
				linestyle="-",
				linewidth=2.8,
				label="Batch",
			)
			ax.plot(
				# Markers on top of the batch line make tiny mismatches visually obvious.
				serial_xy[:, 0],
				serial_xy[:, 1],
				color="black",
				linestyle="None",
				marker="o",
				markersize=2.6,
				markevery=max(1, len(serial_xy) // 25),
				label="Serial (markers)",
			)

			ax.set_title(f"{framework_name} · {label}\nallclose={is_match}, max|Δ|={max_diff:.2e}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.grid(alpha=0.25)
			ax.axis("equal")
			ax.legend(frameon=False)

	fig.savefig(out_path, dpi=180)
	print(f"Saved plot to: {out_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Demonstrate JAX vs PyTorch RNG reproducibility.")
	parser.add_argument("--steps", type=int, default=300, help="Number of random-walk steps.")
	parser.add_argument("--seed", type=int, default=7, help="Random seed.")
	parser.add_argument("--step-std", type=float, default=0.2, help="Standard deviation per step.")
	parser.add_argument("--out", type=str, default="extra_project/jax_vs_torch_random_walks.png", help="Output plot path.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	try:
		jax_result = simulate_jax(args.steps, args.seed, args.step_std)
	except Exception as exc:
		raise RuntimeError(
			"JAX simulation failed. Make sure `jax` is installed in the active environment."
		) from exc

	try:
		torch_result = simulate_torch(args.steps, args.seed, args.step_std)
	except Exception as exc:
		raise RuntimeError(
			"PyTorch simulation failed. Make sure `torch` is installed in the active environment."
		) from exc

	jax_diff_a, jax_diff_b = max_abs_diff(jax_result.batch, jax_result.serial)
	torch_diff_a, torch_diff_b = max_abs_diff(torch_result.batch, torch_result.serial)

	print("\nMax |Batch - Serial| per particle")
	print(f"JAX      -> A: {jax_diff_a:.3e}, B: {jax_diff_b:.3e}")
	print(f"PyTorch  -> A: {torch_diff_a:.3e}, B: {torch_diff_b:.3e}")

	print("\nInterpretation:")
	print("- In JAX, explicit PRNG keys remove hidden coupling; trajectories are invariant to batching.")
	print("- In PyTorch, a single global RNG stream means draw order changes with batching,")
	print("  so trajectories can depend on unrelated particles/processes.")
	print("- This hidden coupling is risky for scientific reproducibility, distributed runs,")
	print("  and Neural SDE-style stochastic simulations.")

	plot_comparison(jax_result, torch_result, args.out)


if __name__ == "__main__":
	main()

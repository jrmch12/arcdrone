#!/usr/bin/env python3
"""Benchmark jax_gsplat rendering throughput.

Measures renders/second for various batch sizes and resolutions,
using either the real scene.ply or a synthetic Gaussian cloud.

Usage:
    conda run -n arcdrone_gs python tools/benchmark_gsplat.py
    conda run -n arcdrone_gs python tools/benchmark_gsplat.py --scene assets/gs/scene.ply
    conda run -n arcdrone_gs python tools/benchmark_gsplat.py --synthetic 50000
"""

import argparse
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np


def make_synthetic_scene(n_gaussians: int):
    """Create a synthetic GSScene with random Gaussians for benchmarking."""
    from jax_gsplat import GSScene

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    return GSScene(
        means3d=jax.random.normal(k1, (n_gaussians, 3)).astype(jnp.float32) * 2.0,
        scales=jnp.ones((n_gaussians, 3), dtype=jnp.float32) * 0.02,
        quats=jnp.tile(jnp.array([1.0, 0, 0, 0], dtype=jnp.float32), (n_gaussians, 1)),
        colors=jax.random.uniform(k2, (n_gaussians, 3)).astype(jnp.float32),
        opacities=jnp.ones(n_gaussians, dtype=jnp.float32) * 0.8,
    )


def make_random_viewmats(batch_size: int, key):
    """Generate random but valid view matrices (camera looking roughly toward origin)."""
    keys = jax.random.split(key, batch_size)

    def _make_one(k):
        k1, k2 = jax.random.split(k)
        pos = jax.random.normal(k1, (3,)) * 3.0
        pos = pos.at[2].add(3.0)

        forward = -pos / (jnp.linalg.norm(pos) + 1e-8)
        world_up = jnp.array([0.0, 0.0, 1.0])
        right = jnp.cross(forward, world_up)
        right = right / (jnp.linalg.norm(right) + 1e-8)
        up = jnp.cross(right, forward)

        R = jnp.stack([right, -up, forward], axis=0)
        t = -R @ pos

        viewmat = jnp.eye(4, dtype=jnp.float32)
        viewmat = viewmat.at[:3, :3].set(R)
        viewmat = viewmat.at[:3, 3].set(t)
        return viewmat

    return jax.vmap(_make_one)(keys)


def benchmark_render(scene, batch_sizes, resolutions, n_warmup=3, n_iters=20):
    """Run rendering benchmarks across batch sizes and resolutions."""
    from jax_gsplat import render

    n_gaussians = scene.means3d.shape[0]
    results = []

    for H, W in resolutions:
        for B in batch_sizes:
            label = f"B={B:>4d}, {H}x{W}"
            try:
                viewmats = make_random_viewmats(B, jax.random.PRNGKey(B))

                # Warmup: compile + run a few times
                for _ in range(n_warmup):
                    out = render(scene, viewmats, img_shape=(H, W))
                    out.block_until_ready()

                # Timed runs
                times = []
                for _ in range(n_iters):
                    t0 = time.perf_counter()
                    out = render(scene, viewmats, img_shape=(H, W))
                    out.block_until_ready()
                    times.append(time.perf_counter() - t0)

                times = np.array(times)
                mean_ms = times.mean() * 1000
                std_ms = times.std() * 1000
                renders_per_sec = B / times.mean()

                results.append({
                    "batch": B,
                    "resolution": f"{H}x{W}",
                    "mean_ms": mean_ms,
                    "std_ms": std_ms,
                    "renders_per_sec": renders_per_sec,
                    "total_pixels_per_sec": renders_per_sec * H * W,
                })

                print(f"  {label}:  {mean_ms:8.2f} ± {std_ms:5.2f} ms  |  "
                      f"{renders_per_sec:10.1f} renders/s  |  "
                      f"{renders_per_sec * H * W / 1e6:8.2f} Mpix/s")

            except Exception as e:
                print(f"  {label}:  FAILED — {e}")
                results.append({
                    "batch": B,
                    "resolution": f"{H}x{W}",
                    "error": str(e),
                })

    return results


def estimate_rtx4090(results_1050ti):
    """Rough extrapolation from 1050 Ti to RTX 4090.

    RTX 4090: 16384 CUDA cores, 1008 GB/s bandwidth, 82.6 TFLOPS FP32
    GTX 1050 Ti: 768 CUDA cores, 112 GB/s bandwidth, 2.1 TFLOPS FP32
    Conservative multiplier: ~15-25x (memory-bound workloads ~9x, compute-bound ~39x)
    We use ~15x as a conservative estimate since splatting is memory-heavy.
    """
    MULTIPLIER = 15.0
    print(f"\n  (Using {MULTIPLIER:.0f}x conservative multiplier: "
          f"1050Ti={2.1:.1f} TFLOPS / 112 GB/s → 4090={82.6:.1f} TFLOPS / 1008 GB/s)")
    print()

    for r in results_1050ti:
        if "error" in r:
            continue
        est_rps = r["renders_per_sec"] * MULTIPLIER
        est_ms = r["mean_ms"] / MULTIPLIER
        print(f"  B={r['batch']:>4d}, {r['resolution']}:  "
              f"~{est_ms:7.2f} ms  |  ~{est_rps:10.0f} renders/s  |  "
              f"~{est_rps * int(r['resolution'].split('x')[0]) * int(r['resolution'].split('x')[1]) / 1e6:8.1f} Mpix/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark jax_gsplat renderer")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to .ply scene file (default: use synthetic)")
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Number of synthetic Gaussians (default: 50000)")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 4, 16, 64],
                        help="Batch sizes to test")
    parser.add_argument("--resolutions", type=str, nargs="+",
                        default=["64x64", "128x128", "256x256"],
                        help="Resolutions to test (HxW)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    resolutions = []
    for r in args.resolutions:
        h, w = r.split("x")
        resolutions.append((int(h), int(w)))

    # GPU info
    print("=" * 80)
    print("jax_gsplat Rendering Benchmark")
    print("=" * 80)
    devices = jax.devices()
    print(f"  JAX version:  {jax.__version__}")
    print(f"  Devices:      {devices}")
    print(f"  Backend:      {jax.default_backend()}")
    print()

    # Load scene
    if args.scene:
        from jax_gsplat import load_ply
        print(f"Loading scene: {args.scene}")
        t0 = time.perf_counter()
        scene = load_ply(args.scene)
        load_time = time.perf_counter() - t0
        n_gaussians = scene.means3d.shape[0]
        print(f"  {n_gaussians:,} Gaussians loaded in {load_time:.1f}s")
        mem_mb = sum(a.nbytes for a in scene) / 1e6
        print(f"  Scene GPU memory: {mem_mb:.1f} MB")
    else:
        n_synth = args.synthetic or 50000
        print(f"Using synthetic scene: {n_synth:,} Gaussians")
        scene = make_synthetic_scene(n_synth)
        n_gaussians = n_synth

    print(f"\nBenchmark config:")
    print(f"  Gaussians:    {n_gaussians:,}")
    print(f"  Batch sizes:  {args.batch_sizes}")
    print(f"  Resolutions:  {[f'{h}x{w}' for h, w in resolutions]}")
    print(f"  Warmup:       {args.warmup}")
    print(f"  Iterations:   {args.iters}")

    # Run benchmarks
    print(f"\n{'─' * 80}")
    print("Results (lower ms = better, higher renders/s = better):")
    print(f"{'─' * 80}")

    results = benchmark_render(
        scene, args.batch_sizes, resolutions,
        n_warmup=args.warmup, n_iters=args.iters,
    )

    # RTX 4090 estimate
    valid = [r for r in results if "error" not in r]
    if valid:
        print(f"\n{'─' * 80}")
        print("Estimated RTX 4090 performance (conservative):")
        print(f"{'─' * 80}")
        estimate_rtx4090(valid)

        # Summary for RL training context
        print(f"\n{'─' * 80}")
        print("RL Training Context:")
        print(f"{'─' * 80}")
        best_64 = [r for r in valid if r["resolution"] == "64x64"]
        if best_64:
            best = max(best_64, key=lambda r: r["renders_per_sec"])
            print(f"  Best 64x64 throughput (this GPU):  {best['renders_per_sec']:.0f} renders/s "
                  f"@ B={best['batch']}")
            est_4090 = best["renders_per_sec"] * 15
            print(f"  Est. RTX 4090 throughput:          ~{est_4090:.0f} renders/s")
            print(f"  gauss_gym reference (4090):        ~100,000+ steps/s")
            print(f"  Note: each training step = 1 render, so renders/s ≈ steps/s")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()

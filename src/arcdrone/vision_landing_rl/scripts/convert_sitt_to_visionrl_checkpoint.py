"""Build a vision-RL warmstart checkpoint from a SITT run folder.

Expected SITT artifacts in a run directory:
  - teacher_model.pkl
  - student_model.pkl
  - proxy_model.pkl (optional, ignored)

Output format (ready for vision_landing_rl/train.py restore path):
  (normalizer_params, policy_params, value_params)

Where:
  - policy_params come from student_model.pkl
  - value_params come from teacher_model.pkl
  - normalizer_params merges:
      * proprio_obs stats from student normalizer
      * value_obs stats from teacher normalizer
      * teacher_obs stats from teacher value stats (same as value_obs)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from brax.io import model


def _resolve_run_dir(path_or_dir: str) -> Path:
    p = Path(path_or_dir)
    return p if p.is_dir() else p.parent


def _find_first_mapping_key(tree: Any, candidates: list[str]) -> str | None:
    if not isinstance(tree, Mapping):
        return None
    for key in candidates:
        if key in tree:
            return key
    return None


def _extract_norm_leaf(norm_state: Any, candidates: list[str], label: str):
    """Extract mean/std/summed_variance leaves from a RunningStatisticsState.

    Supports both flat and dict-backed normalizers.
    """
    mean = getattr(norm_state, "mean")
    std = getattr(norm_state, "std")
    summed_variance = getattr(norm_state, "summed_variance")

    if isinstance(mean, Mapping):
        key = _find_first_mapping_key(mean, candidates)
        if key is None:
            raise KeyError(
                f"Could not find any of {candidates} in {label} normalizer keys: {list(mean.keys())}"
            )
        return mean[key], std[key], summed_variance[key], key

    # Flat normalizer.
    return mean, std, summed_variance, "<flat>"


def build_checkpoint(
    run_dir: Path,
    teacher_file: str,
    student_file: str,
    proxy_file: str,
    output_file: str,
) -> Path:
    teacher_path = run_dir / teacher_file
    student_path = run_dir / student_file
    proxy_path = run_dir / proxy_file

    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Missing student checkpoint: {student_path}")

    teacher_ckpt = model.load_params(str(teacher_path))
    student_ckpt = model.load_params(str(student_path))

    if proxy_path.exists():
        print(f"[Info] Found proxy checkpoint (unused for conversion): {proxy_path}")
    else:
        print("[Info] proxy_model.pkl not found; continuing without it.")

    # Teacher format: (teacher_norm, teacher_policy, teacher_value, ...optional extras)
    if not isinstance(teacher_ckpt, tuple) or len(teacher_ckpt) < 3:
        raise ValueError(
            "teacher_model.pkl must be a tuple with at least 3 entries: (norm, policy, value, ...)"
        )
    teacher_norm = teacher_ckpt[0]
    teacher_value_params = teacher_ckpt[2]

    # Student format: (student_norm, (student_enc, action_head))
    if (
        not isinstance(student_ckpt, tuple)
        or len(student_ckpt) != 2
        or not isinstance(student_ckpt[1], tuple)
        or len(student_ckpt[1]) < 2
    ):
        raise ValueError(
            "student_model.pkl must have format: (student_norm, (student_enc, action_head))"
        )
    student_norm = student_ckpt[0]
    student_policy_params = student_ckpt[1]

    student_mean, student_std, student_sv, student_key = _extract_norm_leaf(
        student_norm,
        candidates=["proprio_obs", "proprio", "policy_obs"],
        label="student",
    )
    teacher_mean, teacher_std, teacher_sv, teacher_key = _extract_norm_leaf(
        teacher_norm,
        candidates=["value_obs", "teacher_obs", "policy_obs"],
        label="teacher",
    )

    merged_mean = {
        "proprio_obs": student_mean,
        "value_obs": teacher_mean,
        "teacher_obs": teacher_mean,
    }
    merged_std = {
        "proprio_obs": student_std,
        "value_obs": teacher_std,
        "teacher_obs": teacher_std,
    }
    merged_summed_variance = {
        "proprio_obs": student_sv,
        "value_obs": teacher_sv,
        "teacher_obs": teacher_sv,
    }

    if not hasattr(teacher_norm, "replace"):
        raise TypeError(
            "Unsupported teacher normalizer type; expected a flax struct with .replace()."
        )

    merged_norm = teacher_norm.replace(
        mean=merged_mean,
        std=merged_std,
        summed_variance=merged_summed_variance,
    )

    visionrl_ckpt = (
        merged_norm,
        student_policy_params,
        teacher_value_params,
    )

    output_path = run_dir / output_file
    model.save_params(str(output_path), visionrl_ckpt)

    print(f"[OK] VisionRL warmstart checkpoint saved: {output_path}")
    print(f"[Info] Student norm key used: {student_key}")
    print(f"[Info] Teacher norm key used: {teacher_key}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SITT outputs into a VisionRL restore checkpoint."
    )
    parser.add_argument(
        "run_path",
        help="Path to SITT run directory or any file inside it.",
    )
    parser.add_argument(
        "--teacher-file",
        default="teacher_model.pkl",
        help="Teacher checkpoint filename inside run dir.",
    )
    parser.add_argument(
        "--student-file",
        default="student_model.pkl",
        help="Student checkpoint filename inside run dir.",
    )
    parser.add_argument(
        "--proxy-file",
        default="proxy_model.pkl",
        help="Proxy checkpoint filename inside run dir (optional, ignored).",
    )
    parser.add_argument(
        "--output-file",
        default="visionrl_warmstart.pkl",
        help="Output filename inside run dir.",
    )

    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_path)
    build_checkpoint(
        run_dir=run_dir,
        teacher_file=args.teacher_file,
        student_file=args.student_file,
        proxy_file=args.proxy_file,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

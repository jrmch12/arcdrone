#!/usr/bin/env python3
"""
Compose and print the project's Hydra config as a resolved Python dict.
Usage examples (run from repository root):

# default composition (uses defaults in config.yaml)
python tools/print_hydra_cfg.py

# with overrides
python tools/print_hydra_cfg.py task=hover train.num_envs=2056

You can pass any Hydra override as CLI args.
"""
import sys
from pprint import pprint

from hydra import initialize, compose
try:
    # Hydra exposes an API that accepts absolute paths for config dirs
    from hydra.experimental import initialize_config_dir
except Exception:
    initialize_config_dir = None
from omegaconf import OmegaConf
from pathlib import Path
import os
import traceback
from hydra.errors import MissingConfigException


def main(overrides):
    # change cwd to repo root so Hydra accepts a relative config_path
    repo_root = Path(__file__).resolve().parent.parent
    cfg_rel = "src/arcdrone/controller/cfg"
    cfg_dir = (repo_root / cfg_rel).resolve()
    cwd_before = Path.cwd()
    # chdir and show debug info
    os.chdir(str(repo_root))
    print(f"[debug] cwd after chdir: {os.getcwd()}")
    print(f"[debug] repo_root: {repo_root}")
    print(f"[debug] config relative path passed to hydra.initialize: {cfg_rel}")
    print(f"[debug] absolute config path: {cfg_dir}")
    print(f"[debug] script file dir: {Path(__file__).resolve().parent}")

    if not cfg_dir.is_dir():
        print("[error] computed config directory does not exist:", cfg_dir)
        sys.exit(1)

    # Ensure the 'train' defaults reference exists; if not, add a sensible override.
    # Determine if caller provided explicit 'train' or 'task' overrides.
    override_keys = {o.split("=", 1)[0].split(":", 1)[0] for o in overrides}
    if "train" not in override_keys:
        # find effective task: from overrides if provided, otherwise from config.yaml defaults
        task_name = None
        for o in overrides:
            if o.startswith("task=") or o.startswith("task:"):
                task_name = o.split("=", 1)[1] if "=" in o else o.split(":", 1)[1]
                break
        if task_name is None:
            try:
                text = (cfg_dir / "config.yaml").read_text()
                import re

                m = re.search(r"^\s*-\s*task:\s*(\S+)", text, re.MULTILINE)
                if m:
                    task_name = m.group(1).strip().strip('"\'')
            except Exception:
                task_name = None

        if task_name:
            # available train options
            train_dir = cfg_dir / "train"
            try:
                available = {p.stem for p in train_dir.iterdir() if p.is_file()}
            except Exception:
                available = set()

            candidates = [f"{task_name}_ppo", f"{task_name}_sitt", "default_ppo"]
            chosen = None
            for c in candidates:
                if c in available:
                    chosen = c
                    break
            if chosen:
                print(f"[debug] adding override train={chosen} (auto-selected)")
                overrides = overrides + [f"train={chosen}"]

    try:
        try:
            # Prefer using initialize_config_dir when available (accepts absolute paths)
            if initialize_config_dir is not None:
                print("[debug] using hydra.experimental.initialize_config_dir with absolute path")
                # initialize_config_dir may not accept the same args as initialize(); call with minimal args
                try:
                    with initialize_config_dir(config_dir=str(cfg_dir)):
                        cfg = compose(config_name="config", overrides=overrides)
                except TypeError:
                    # fallback: call without context manager if signature differs
                    initialize_config_dir(config_dir=str(cfg_dir))
                    try:
                        cfg = compose(config_name="config", overrides=overrides)
                    finally:
                        # no explicit cleanup API available; rely on process exit
                        pass
            else:
                # Pass job_name to avoid hydra using script's directory as base
                print("[debug] using hydra.initialize with relative path and job_name")
                with initialize(config_path=cfg_rel, job_name="print_hydra_cfg", version_base=None):
                    cfg = compose(config_name="config", overrides=overrides)
            pprint(OmegaConf.to_container(cfg, resolve=True))
        except MissingConfigException as e:
            attempted = os.path.join(os.getcwd(), cfg_rel)
            print("[error] Hydra could not find the primary config directory.")
            print(f"[error] attempted path: {attempted}")
            print("[error] Directory listing of parent:")
            parent = os.path.dirname(attempted)
            try:
                for p in os.listdir(parent):
                    print(" - ", p)
            except Exception:
                print("[error] could not list parent directory contents")
            print("Original Hydra error:")
            traceback.print_exc()
            raise
    finally:
        os.chdir(str(cwd_before))


if __name__ == "__main__":
    overrides = sys.argv[1:]
    main(overrides)

# How to use
# python tools/print_hydra_cfg.py task=hover train.num_envs=2056 train.unroll_length=32
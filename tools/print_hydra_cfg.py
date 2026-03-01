import sys
from pathlib import Path
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "src/arcdrone/controller/cfg"

def main():
    initialize_config_dir(config_dir=str(CFG_DIR), job_name="print_cfg", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    print(OmegaConf.to_yaml(cfg))

if __name__ == '__main__':
    main()

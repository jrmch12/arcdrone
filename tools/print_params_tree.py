import argparse
from brax.io import model


def describe(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, tuple):
        print(f"{prefix}tuple (len={len(obj)})")
        for i, v in enumerate(obj):
            print(f"{prefix}  [{i}]:")
            describe(v, indent + 2)
    elif isinstance(obj, dict):
        print(f"{prefix}dict keys={list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix}  '{k}':")
            describe(v, indent + 2)
    else:
        info = getattr(obj, "shape", repr(obj))
        print(f"{prefix}{type(obj).__name__}: {info}")


def main():
    parser = argparse.ArgumentParser(
        description="Describe the structure of a Brax checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        type=str,
        help="Path to the .pkl checkpoint to describe.",
    )
    args = parser.parse_args()
    params = model.load_params(args.checkpoint)
    describe(params)


if __name__ == "__main__":
    main()

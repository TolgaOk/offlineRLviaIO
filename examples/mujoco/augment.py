"""Script to create augmented datasets with different feature configurations."""
import argparse
from io_agent.plant.mujoco import HopperEnv, Walker2dEnv, HalfCheetahEnv
from utils import augment_mujoco_dataset

registered_envs = dict(
    walker2d=Walker2dEnv,
    hopper=HopperEnv,
    cheetah=HalfCheetahEnv
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--name", type=str, required=True,
                        help="Name for this augmentation config (e.g., 'default', 'rff128')")
    parser.add_argument("--save-dir", type=str, default="./data")
    # Feature options (use BooleanOptionalAction: --use-xxx or --no-use-xxx)
    parser.add_argument("--n-past", type=int, default=4)
    parser.add_argument("--use-co-product", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sinusoidal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-state-regressor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-action-regressor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rff-n-features", type=int, default=1024)
    parser.add_argument("--rff-sigma", type=float, default=0.1)

    args = parser.parse_args()

    if args.env_name not in registered_envs:
        raise ValueError(f"Unknown env: {args.env_name}. Choose from {list(registered_envs.keys())}")

    env = registered_envs[args.env_name]()
    save_dir = f"{args.save_dir}/{args.env_name}"

    data_path = augment_mujoco_dataset(
        env=env,
        save_dir=save_dir,
        name=args.name,
        n_past=args.n_past,
        use_co_product=args.use_co_product,
        use_sinusoidal=args.use_sinusoidal,
        use_state_regressor=args.use_state_regressor,
        use_action_regressor=args.use_action_regressor,
        use_rff=args.use_rff,
        rff_n_features=args.rff_n_features,
        rff_sigma=args.rff_sigma,
    )
    print(f"Created: {data_path}")

from model import MuZero, optimizer as make_optimizer
from nn import _init_representation_func, _init_prediction_func, _init_dynamic_func, Representation, Prediction, Dynamic
from train_cartpole_purejax_fit import fit_pure_cartpole
import wandb
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support_size", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=8)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--k_steps", type=int, default=10)
    parser.add_argument("--wandb_project", type=str, default="muax_jax_cartpole")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--max_episodes", type=int, default=2000)
    args = parser.parse_args()
    wandb.init(project=args.wandb_project, name=f"num_simulations_{args.num_simulations}_episodes_{args.max_episodes}", mode=args.wandb_mode)
    support_size = args.support_size
    embedding_size = args.embedding_size
    discount = args.discount
    num_actions = args.num_actions
    num_simulations = args.num_simulations
    k_steps = args.k_steps
    full_support_size = support_size * 2 + 1
    max_episodes = args.max_episodes
    wandb.config.update({
        "support_size": support_size,
        "embedding_size": embedding_size,
        "discount": discount,
        "num_actions": num_actions,
        "full_support_size": full_support_size,
        "num_simulations": num_simulations,
        "k_steps": k_steps,
        "max_episodes": max_episodes,
    })
    repr_fn = _init_representation_func(Representation, embedding_size)
    pred_fn = _init_prediction_func(Prediction, num_actions, full_support_size)
    dy_fn   = _init_dynamic_func(Dynamic, embedding_size, num_actions, full_support_size)

    gradient_transform = make_optimizer(
        init_value=0.02,
        peak_value=0.02,
        end_value=0.002,
        warmup_steps=5000,
        transition_steps=5000,
    )

    model = MuZero(
        repr_fn,
        pred_fn,
        dy_fn,
        policy="muzero",
        discount=discount,
        optimizer=gradient_transform,
        support_size=support_size,
    )

    trained_model = fit_pure_cartpole(model, num_simulations=num_simulations, k_steps=k_steps, max_episodes=max_episodes)
    wandb.finish()

if __name__ == "__main__":
    main()

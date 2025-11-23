from model import MuZero, optimizer as make_optimizer
from nn import _init_representation_func, _init_prediction_func, _init_dynamic_func, Representation, Prediction, Dynamic
from train_cartpole_purejax_fit import fit_pure_cartpole
import wandb

def main():
    wandb.init(project="muax_jax_cartpole", name="full_jit_train", mode="online")
    support_size = 10
    embedding_size = 8
    discount = 0.99
    num_actions = 2
    full_support_size = support_size * 2 + 1
    wandb.config.update({
        "support_size": support_size,
        "embedding_size": embedding_size,
        "discount": discount,
        "num_actions": num_actions,
        "full_support_size": full_support_size,
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

    trained_model = fit_pure_cartpole(model)
    wandb.finish()

if __name__ == "__main__":
    main()

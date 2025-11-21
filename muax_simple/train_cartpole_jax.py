from nn import _init_representation_func, _init_prediction_func, _init_dynamic_func, Representation, Prediction, Dynamic
from episode_tracer import PNStep
from replay_buffer import TrajectoryReplayBuffer
from model import MuZero
from model import optimizer
from train import fit
from wrappers import JaxToGymWrapper
from cartpole_jax_env import CartPole

jax_cartpole = CartPole()
train_env = JaxToGymWrapper(jax_cartpole, seed=42)
test_env = JaxToGymWrapper(jax_cartpole, seed=43)
support_size = 10 
embedding_size = 8
discount = 0.99
num_actions = 2
full_support_size = int(support_size * 2 + 1)

repr_fn = _init_representation_func(Representation, embedding_size)
pred_fn = _init_prediction_func(Prediction, num_actions, full_support_size)
dy_fn = _init_dynamic_func(Dynamic, embedding_size, num_actions, full_support_size)

tracer = PNStep(10, discount, 0.5)
buffer = TrajectoryReplayBuffer(500)

gradient_transform = optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)

model = MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                    optimizer=gradient_transform, support_size=support_size)

model_path = fit(model, env_id=None, env=train_env, test_env=test_env, 
                    max_episodes=2000,
                    max_training_steps=400000,
                    tracer=tracer,
                    buffer=buffer,
                    k_steps=10,
                    sample_per_trajectory=1,
                    num_trajectory=32,
                    tensorboard_dir='./content/tensorboard/cartpole',
                    model_save_path='./content/models/cartpole',
                    save_name='cartpole_model_params',
                    random_seed=0,
                    log_all_metrics=True)
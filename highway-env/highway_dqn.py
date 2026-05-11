if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.utils import plot_history

import gymnasium
from gymnasium.wrappers import RecordVideo
import highway_env
from stable_baselines3 import DQN

import torch


video_episode_interval = 500
video_folder = 'highway-env/videos/highway-fast-dqn'
video_name_prefix = 'highway-fast-v0'

env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: (episode_id + 1) % video_episode_interval == 0,
    name_prefix=video_name_prefix,
)
model = DQN(
    'MlpPolicy', 
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    # tensorboard_log="highway_dqn/"
)
model.learn(2e4)
# model.save("highway_dqn/model")
env.close()

# Load and test saved model
# model = DQN.load("highway_dqn/model")

done = truncated = False
eval_env = gymnasium.make("highway-fast-v0", render_mode='rgb_array')
eval_env = RecordVideo(
    eval_env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: episode_id == 0,
    name_prefix=video_name_prefix+"-test-",
)
obs, info = eval_env.reset()
q_history = []
reward_history = []
total_reward_history = []
total_reward = 0
while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]
    # print("Action:", type(action), action)#Action: <class 'numpy.ndarray'> 0や4など
    # print("Q-Values:", type(q_values), q_values)#Q-Values: <class 'numpy.ndarray'> [ 0.12574589  0.06204881  0.09046473 -0.02809356 -0.07172634]
    q = q_values[action]
    # print("Selected Q-Value:", type(q), q)
    obs, reward, done, truncated, info = eval_env.step(action)
    # print("Reward:", type(reward), reward)#Reward: <class 'numpy.float64'> 0.7999999999999999
    # eval_env.render()
    q_history.append(q)
    reward_history.append(reward)
    total_reward += reward
    total_reward_history.append(total_reward)

plot_history(q_history, ylabel="Q-Value", fig_name=f'{video_folder}/{video_name_prefix}-test-q_value.png')
plot_history(reward_history, ylabel="Reward", fig_name=f'{video_folder}/{video_name_prefix}-test-reward.png')
plot_history(total_reward_history, ylabel="Total Reward", fig_name=f'{video_folder}/{video_name_prefix}-test-total_reward_dqn.png')

eval_env.close()

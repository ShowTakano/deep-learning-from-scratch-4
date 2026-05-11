if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.utils import plot_history
import datetime


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x), dim=1)
        return x


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :])
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(probs)
        action = m.sample().item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        state = torch.tensor(state[np.newaxis, :])
        next_state = torch.tensor(next_state[np.newaxis, :])

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()
        return v, delta, loss_v, loss_pi

episodes = 3000
video_episode_interval = 500
video_folder = 'videos'
video_name_prefix = 'ac-cartpole'
env = gym.make('CartPole-v1', render_mode='rgb_array')
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda episode_id: (episode_id + 1) % video_episode_interval == 0,
    name_prefix=video_name_prefix,
)
agent = Agent()
total_reward_history = []

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    v_history = []
    delta_history = []
    loss_v_history = []
    loss_pi_history = []
    reward_history = []

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        v, delta, loss_v, loss_pi = agent.update(state, prob, reward, next_state, done)
        v_history.append(v[0].detach().cpu().numpy())
        delta_history.append(delta[0].detach().cpu().numpy())
        loss_v_history.append(loss_v.detach().cpu().numpy())
        loss_pi_history.append(loss_pi.detach().cpu().numpy())
        reward_history.append(reward)

        state = next_state
        total_reward += reward

    total_reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}, loss_v : {:.4f}, loss_pi : {:.4f}".format(episode, total_reward, loss_v, loss_pi))
    if episode != 0 and episode % video_episode_interval == 0:
        # save image
        e_idx = episode -1
        plot_history(v_history, ylabel="Value", fig_name=f'{video_folder}/{video_name_prefix}-episode-{e_idx}-v.png')
        plot_history(delta_history, ylabel="Delta", fig_name=f'{video_folder}/{video_name_prefix}-episode-{e_idx}-delta.png')
        plot_history(loss_v_history, ylabel="Loss V", fig_name=f'{video_folder}/{video_name_prefix}-episode-{e_idx}-loss_v.png')
        plot_history(loss_pi_history, ylabel="Loss Pi", fig_name=f'{video_folder}/{video_name_prefix}-episode-{e_idx}-loss_pi.png')
        plot_history(reward_history, ylabel="Reward", fig_name=f'{video_folder}/{video_name_prefix}-episode-{e_idx}-reward.png')

plot_history(
    total_reward_history, 
    ylabel="Total Reward", 
    fig_name=f'{video_folder}/{video_name_prefix}-total_reward_actor_critic_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
)
# save the model
# torch.save(agent.pi.state_dict(), 'policy_net.pth')
# torch.save(agent.v.state_dict(), 'value_net.pth')
env.close()

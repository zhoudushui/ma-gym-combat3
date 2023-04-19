import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from ma_gym.envs.combat3.combat3 import Combat3


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

class PPO:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device


    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        log_probs = torch.log(self.actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

actor_lr = 3e-4
# critic_lr = 1e-3
critic_lr = 1e-4
num_episodes = 20000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
device = torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

team_size = 5
grid_size = (15, 15)
# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为5
env = Combat3(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size, n_neutrals=team_size)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_list = []

            for j in range(team_size):
                transition_list.append({
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            })

            s = env.reset()
            terminal = False
            while not terminal:
                a = []
                for k in range(team_size):
                    a.append(agent.take_action(s[k]))
                next_s, r, done, info = env.step(a)
                for k in range(team_size):
                    transition_list[k]['states'].append(s[k])
                    transition_list[k]['actions'].append(a[k])
                    transition_list[k]['next_states'].append(next_s[k])
                    transition_list[k]['rewards'].append(
                        r[k] + 100 if info['win'] else r[k] - 0.5)
                    transition_list[k]['dones'].append(False)
                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            for k in range(team_size):
                agent.update(transition_list[k])
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
win_array = np.array(win_list)
# 每100条轨迹取一次平均
win_array = np.mean(win_array.reshape(-1, 100), axis=1)

episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Win rate')
plt.title('IPPO on Combat')
plt.show()

win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # transition_list = []
            #
            # for j in range(team_size):
            #     transition_list.append({
            #     'states': [],
            #     'actions': [],
            #     'next_states': [],
            #     'rewards': [],
            #     'dones': []
            # })

            s = env.reset()
            terminal = False
            while not terminal:
                a = []
                for k in range(team_size):
                    a.append(agent.take_action(s[k]))
                next_s, r, done, info = env.step(a)
                # for k in range(team_size):
                #     transition_list[k]['states'].append(s[k])
                #     transition_list[k]['actions'].append(a[k])
                #     transition_list[k]['next_states'].append(next_s[k])
                #     transition_list[k]['rewards'].append(
                #         r[k] + 100 if info['win'] else r[k] - 0.5)
                #     transition_list[k]['dones'].append(False)
                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            # for k in range(team_size):
            #     agent.update(transition_list[k])
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
win_array = np.array(win_list)
# 每100条轨迹取一次平均
win_array = np.mean(win_array.reshape(-1, 100), axis=1)

episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Win rate')
plt.title('Trained Combat')
plt.show()
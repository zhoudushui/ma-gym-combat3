import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from ma_gym.envs.combat.combat import Combat
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

class MAPPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, eps, gamma, num_workers, device):
        self.actor_nets = [PolicyNet(state_dim, hidden_dim, action_dim).to(device) for _ in range(num_workers)]
        self.critic_nets = [ValueNet(state_dim, hidden_dim).to(device) for _ in range(num_workers)]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actor_nets]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critic_nets]
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.num_workers = num_workers
        self.device = device


    def take_action(self, state, worker_idx):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor_nets[worker_idx](state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


    def update(self, transition_dicts):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        for worker_idx in range(self.num_workers):
            transition_dict = transition_dicts[worker_idx]
            states = torch.tensor(transition_dict['states'],
                                  dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'],
                                 dtype=torch.float).view(-1, 1).to(self.device)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_dones.append(dones)

        all_states = torch.cat(all_states)
        all_actions = torch.cat(all_actions)
        all_rewards = torch.cat(all_rewards)
        all_next_states = torch.cat(all_next_states)
        all_dones = torch.cat(all_dones)

        with torch.no_grad():
            all_old_log_probs = torch.stack(
                [torch.log(actor(all_states).gather(1, all_actions)).squeeze() for actor in self.actor_nets], dim=1)

            all_td_target = all_rewards + self.gamma * torch.stack(
                [critic(all_next_states).squeeze() * (1 - all_dones) for critic in self.critic_nets], dim=1).mean(dim=1,
                                                                                                                  keepdim=True)

            all_td_delta = all_td_target - torch.stack([critic(all_states).squeeze() for critic in self.critic_nets], dim=1).mean(dim=1,
                                                                                                                  keepdim=True)

            all_advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, all_td_delta.cpu()).to(self.device)

            all_old_log_probs = torch.stack(
                [torch.log(actor(all_states).gather(1, all_actions)).squeeze() for actor in self.actor_nets], dim=1)

        for i in range(len(self.actor_nets)):
            log_probs = torch.log(self.actor_nets[i](all_states).gather(1, all_actions))
            ratio = torch.exp(log_probs - all_old_log_probs[:, i])
            surr1 = ratio * all_advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * all_advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(F.mse_loss(self.critic_nets[i](all_states), all_td_target.detach()))

            self.actor_optimizers[i].zero_grad()
            self.critic_optimizers[i].zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizers[i].step()
            self.critic_optimizers[i].step()

            # 计算所有梯度的平均值，并使用这个平均值更新全局参数
        for param in self.actor_nets[0].parameters():
            param.grad = torch.zeros_like(param.data)
        for param in self.critic_nets[0].parameters():
            param.grad = torch.zeros_like(param.data)

        for i in range(len(self.actor_nets)):
            for param, shared_param in zip(self.actor_nets[i].parameters(), self.actor_nets[0].parameters()):
                shared_param.grad += param.grad / self.num_workers
            for param, shared_param in zip(self.critic_nets[i].parameters(), self.critic_nets[0].parameters()):
                shared_param.grad += param.grad / self.num_workers

        self.actor_optimizers[0].step()
        self.critic_optimizers[0].step()

actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 150000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
# device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# team_size = 5
# grid_size = (15, 15)
# # 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
# env = Combat3(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size, n_neutrals=team_size)
team_size = 3
grid_size = (15, 15)
# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
agent = MAPPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma,team_size, device)

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
                    a.append(agent.take_action(s[k], k))
                next_s, r, done, info = env.step(a)
                for k in range(team_size):
                    transition_list[k]['states'].append(s[k])
                    transition_list[k]['actions'].append(a[k])
                    transition_list[k]['next_states'].append(next_s[k])
                    transition_list[k]['rewards'].append(
                        r[k] + 100 if info['win'] else r[k] - 0.1)
                    transition_list[k]['dones'].append(False)
                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            # for k in range(team_size):
            #     agent.update(transition_list[k])
            agent.update(transition_list)
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
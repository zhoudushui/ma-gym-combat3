import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from ma_gym.envs.combat3.combat3 import Combat3

'''这段代码定义了一个名为PolicyNet的神经网络模型类，该类继承自torch.nn.Module。

__init__()方法中，定义了三个全连接层，分别是输入层、一个隐藏层和输出层，并将它们作为类的编码器(encoder)。
输入层的大小为state_dim，隐藏层的大小为hidden_dim，输出层的大小为action_dim。

在forward()方法中，输入x作为输入层的输入，经过线性变换得到hidden_dim维向量，然后通过ReLU激活函数进行非线性变换；
再将该向量经过线性变换得到action_dim维向量，最后使用softmax函数计算输出层的输出。返回softmax函数的值。

总体来说，这个模型具有一个简单的结构，它接收一个状态，通过两个全连接层进行线性变换，最后通过softmax函数得到输出。
在深度强化学习中，softmax函数的输出通常表示下一步需要采取的行动，因此这个模型可以用于确定策略(policy)。'''

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)

'''该代码定义了一个名为ValueNet的torch神经网络模型类，包括初始化函数和前向传播函数。

初始化函数__init__(self, state_dim, hidden_dim)包括两个参数：状态的维度state_dim和隐藏层的维度hidden_dim。
在该函数中，首先调用父类的初始化函数super(ValueNet, self).__init__()，然后定义了三个线性层：
self.fc1 = torch.nn.Linear(state_dim, hidden_dim)表示输入层到隐藏层的线性映射，
self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)表示隐藏层到隐藏层的线性映射，
self.fc3 = torch.nn.Linear(hidden_dim, 1)表示隐藏层到输出层的线性映射。

前向传播函数forward(self, x)的输入参数为x，即状态，该函数返回该状态对应的值。
在该函数中，首先将输入x通过第一层线性层self.fc1进行线性变换，
再通过relu激活函数非线性变换，再经过第二层线性层self.fc2进行非线性线性变换，
最后通过第三层线性层self.fc3输出最终结果。
即x = F.relu(self.fc2(F.relu(self.fc1(x))))和return self.fc3(x)。
其中F.relu表示relu激活函数。该函数就是神经网络正向传播的过程，将输入x经过多次非线性变换得到最终输出结果。'''

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)

import torch
import torch.nn.functional as F

class MAPPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        agent_index = transition_dict['agent_index']

        # Get critic value estimate for next state
        target_critic_output = torch.zeros_like(dones)
        for i in range(len(target_critic_output)):
            if not dones[-i]:
                target_critic_output[-i] = self.critic(next_states[-i]).detach()

        # Calculate TD target for each agent
        td_target = rewards + self.gamma * target_critic_output
        td_error = td_target - self.critic(states)

        # Calculate advantages for each agent
        agent_advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)

        # Calculate old and new log probabilities
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        new_log_probs = torch.log(self.actor(states).gather(1, actions))

        # Calculate importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Calculate clipped surrogate loss for each agent
        surrogate1 = ratio * agent_advantage
        surrogate2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * agent_advantage

        # Calculate actor loss and critic loss
        actor_loss = torch.mean(-torch.min(surrogate1, surrogate2))
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # Perform update of actor and critic weights
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()


actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 10000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
device = torch.device("cpu")
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

team_size = 5
grid_size = (15, 15)
# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat3(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size, n_neutrals=team_size)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
# 两个智能体共享同一个策略
agent = MAPPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_list = []
            # transition_dict = {
            #     'states': [],
            #     'actions': [],
            #     'next_states': [],
            #     'rewards': [],
            #     'dones': []
            # }
            for j in range(team_size):
                transition_list.append({
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': [],
                    'agent_index': j
                })

            s = env.reset()
            terminal = False
            while not terminal:
                a = []
                for k in range(team_size):
                    a.append(agent.take_action(s[k]))
                # a_1 = agent.take_action(s[0])
                # a_2 = agent.take_action(s[1])
                # next_s, r, done, info = env.step([a_1, a_2])
                next_s, r, done, info = env.step(a)
                for k in range(team_size):
                    transition_list[k]['states'].append(s[k])
                    transition_list[k]['actions'].append(a[k])
                    transition_list[k]['next_states'].append(next_s[k])
                    transition_list[k]['rewards'].append(
                        r[k] + 100 if info['win'] else r[k] - 0.1)
                    transition_list[k]['dones'].append(False)
                # transition_dict_1['states'].append(s[0])
                # transition_dict_1['actions'].append(a_1)
                # transition_dict_1['next_states'].append(next_s[0])
                # # transition_dict_1['rewards'].append(r[0])
                # transition_dict_1['rewards'].append(
                #     r[0] + 100 if info['win'] else r[0] - 0.1)
                # transition_dict_1['dones'].append(False)
                #
                # transition_dict_2['states'].append(s[1])
                # transition_dict_2['actions'].append(a_2)
                # transition_dict_2['next_states'].append(next_s[1])
                # # transition_dict_2['rewards'].append(r[1])
                # transition_dict_2['rewards'].append(
                #     r[1] + 100 if info['win'] else r[1] - 0.1)
                # transition_dict_2['dones'].append(False)
                s = next_s
                terminal = all(done)
            win_list.append(1 if info["win"] else 0)
            for k in range(team_size):
                agent.update(transition_list[k])
            # agent.update(transition_dict_1)
            # agent.update(transition_dict_2)
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
plt.title('MAPPO on Combat')
plt.show()
# lst = []
# transition_dict = {
#     'states': [],
#     'actions': [],
#     'next_states': [],
#     'rewards': [],
#     'dones': []
# }
# for i in range(2):
#     lst.append(
#     {
#     'states': [],
#     'actions': [],
#     'next_states': [],
#     'rewards': [],
#     'dones': []
# })
# s = [1, 2, 3.4, 5, 6]
# for k in range(2):
#     lst[k]['states'].append(s[k])
# print(lst[0]['states'])
# print(lst[1]['states'])
# print(transition_dict['states'])


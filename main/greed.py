import torch
import torch.nn.functional as F
import numpy as np
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from ma_gym.envs.combat3.combat3 import Combat3

num_episodes = 10000

team_size = 5
grid_size = (15, 15)
# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat3(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size, n_neutrals=team_size)

win_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            env.reset()
            observation, reward, done, info = env.step(env.agents_same_action)
            while not all(done):
                observation, reward, done, info = env.step(env.agents_same_action)
            win_list.append(1 if info["win"] else 0)
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                        '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
win_array = np.array(win_list)

win_array = np.mean(win_array.reshape(-1, 100), axis=1)

episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Win rate')
plt.title('Greed on Combat')
plt.show()

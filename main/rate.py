import time
from ma_gym.envs.combat.combat import Combat


team_size = 5
grid_size = (15, 15)
win = 0
# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
for i in range(1000):
    env.reset()   # 初始化本场游戏的环境
    # env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    while not all(done):
        # env.render()
        # time.sleep(0.2)
        observation, reward, done, info = env.step(env.action_space.sample())
    if info['win']:
        win += 1
print(win)

env.close()


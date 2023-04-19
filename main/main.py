import gym
import time
import sys

env = gym.make('ma_gym:Combat3-v0')
# 初始化本场游戏的环境
env.reset()
env.render()
observation, reward, done, info = env.step()
while not all(done):
    env.render()
    time.sleep(0.2)
    observation, reward, done, info = env.step()
env.render()
time.sleep(1)
env.close()


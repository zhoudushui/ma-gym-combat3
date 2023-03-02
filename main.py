import gym
import time
import sys

env = gym.make('ma_gym:Combat3-v0')
env.reset()   # 初始化本场游戏的环境

observation, reward, done, info = env.step(env.action_space.sample())

for _ in range (10):
    env.render()
    time.sleep(0.5)
    observation, reward, done, info = env.step(env.action_space.sample())
env.close()

# -*- coding: utf-8 -*-

import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class Combat3(gym.Env):
    """
    We simulate a simple battle involving two opposing teams in a n x n grid.
    Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
    square around the team center, which is picked uniformly in the grid. At each time step, an agent can
    perform one of the following actions: move one cell in one of four directions; attack another agent
    by specifying its ID j (there are m attack actions, each corresponding to one enemy agent); or do
    nothing. If agent A attacks agent B, then B’s health point will be reduced by 1, but only if B is inside
    the firing range of A (its surrounding 3 × 3 area). Agents need one time step of cooling down after
    an attack, during which they cannot attack. All agents start with 3 health points, and die when their
    health reaches 0. A team will win if all agents in the other team die. The simulation ends when one
    team wins, or neither of teams win within 40 time steps (a draw).

    The model controls one team during training, and the other team consist of bots that follow a hardcoded policy.
    The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
    it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
    is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

    When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
    encoding its unique ID, team ID, location, health points and cooldown. A model controlling an agent
    also sees other agents in its visual range (3 × 3 surrounding area). The model gets reward of -1 if the
    team loses or draws at the end of the game. In addition, it also get reward of −0.1 times the total
    health points of the enemy team, which encourages it to attack enemy bots.

    Reference : Learning Multiagent Communication with Backpropagation
    Url : https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(15, 15), n_agents=5, n_neutrals=5, n_opponents=5, init_health=3,
                 full_observable=False,
                 step_cost=0, max_steps=1000, step_cool=1):  # cool代表每次攻击有一轮的冷却时间 将max_steps改为了200
        self._steps_beyond_done = None
        self._step_count = None
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_neutrals = n_neutrals
        self._n_opponents = n_opponents
        self._max_steps = max_steps
        self._step_cool = step_cool + 1
        self._step_cost = step_cost

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5 + self._n_opponents + self.n_neutrals) for _ in range(self.n_agents)])  # 这里还需要修改,
        # 4个移动动作，10个选定敌人并攻击的动作，1个noop（值为4）
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.agent_prev_pos = {_: None for _ in range(self.n_agents)}
        self.opp_pos = {_: None for _ in range(self._n_opponents)}
        self.opp_prev_pos = {_: None for _ in range(self._n_opponents)}
        # 新增
        self.neu_pos = {_: None for _ in range(self.n_neutrals)}
        self.neu_prev_pos = {_: None for _ in range(self.n_neutrals)}
        # 初始化了四个字典，分别存储agent和opponent的当前位置和前一个位置。
        # 其中，agent_pos和opp_pos存储当前位置，agent_prev_pos和opp_prev_pos存储前一个位置。
        # 这些字典的键是从0到n_agents或_n_opponents-1的整数列表，值都是None。

        self._init_health = init_health
        self.agent_health = {_: None for _ in range(self.n_agents)}
        self.opp_health = {_: None for _ in range(self._n_opponents)}
        self.neu_health = {_: None for _ in range(self.n_neutrals)}
        self._agent_dones = [None for _ in range(self.n_agents)]
        self._agent_cool = {_: None for _ in range(self.n_agents)}
        self._agent_cool_step = {_: None for _ in range(self.n_agents)}
        self._opp_cool = {_: None for _ in range(self._n_opponents)}
        self._opp_cool_step = {_: None for _ in range(self._n_opponents)}
        self._neu_cool = {_: None for _ in range(self.n_neutrals)}
        self._neu_cool_step = {_: None for _ in range(self.n_neutrals)}
        self._total_episode_reward = None
        self.viewer = None
        self.full_observable = full_observable

        # 5 * 5 * (type, id, health, cool, x, y)#观察空间

        self._obs_low = np.repeat(np.array([-1., 0., 0., -1., 0., 0.], dtype=np.float32), 5 * 5)
        self._obs_high = np.repeat(np.array([2., n_opponents, init_health, 1., 1., 1.], dtype=np.float32), 5 * 5)
        # 规定了各个观察指标的上下界
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])
        self.seed()

        # For debug only
        self._agents_trace = {_: None for _ in range(self.n_agents)}
        self._opponents_trace = {_: None for _ in range(self._n_opponents)}
        self._neutrals_trace = {_: None for _ in range(self.n_neutrals)}

    # 此函数的功能是获取每个agent的action的含义，若参数agent_i不为空，
    # 则返回指定agent的action含义，若参数agent_i为空，则返回每个agent的action含义。
    # self._agent_cool_step、self._opp_cool、self._opp_cool_step、
    # self._total_episode_reward、self.viewer及self.full_observable为此函数定义时定义的变量，
    # 用于其他功能
    def get_action_meanings(self, agent_i=None):
        action_meaning = []
        for _ in range(self.n_agents):
            meaning = [ACTION_MEANING[i] for i in range(5)]
            meaning += ['Attack Opponent {}'.format(o) for o in range(self._n_opponents)]
            meaning += ['Attack Neutral {}'.format(o) for o in range(self.n_neutrals)]
            # 将敌方编号转化为字符拼接在 Attack Opponent 后面
            action_meaning.append(meaning)
        if agent_i is not None:
            assert isinstance(agent_i, int)
            assert agent_i <= self.n_agents

            return action_meaning[agent_i]
        else:
            return action_meaning

    @staticmethod
    def _one_hot_encoding(i, n):
        x = np.zeros(n)
        x[i] = 1
        return x.tolist()

    def get_agent_obs(self):
        """
        When input to a model, each agent is represented by a set of one-hot binary vectors {i, t, l, h, c}
        encoding its team ID, unique ID, location, health points and cooldown.
        A model controlling an agent also sees other agents in its visual range (5 × 5 surrounding area).
        :return:
        """
        _obs = []  # 存放每个agent的观测信息
        for agent_i in range(self.n_agents):
            # team id , unique id, location(x,y), health, cooldown
            _agent_i_obs = np.zeros((6, 5, 5))
            # 请注意，这里的观测是指在以agent为中心的5 * 5 区域中以上六个指标的信息
            # 即_agent_i_obs[0]中记录的全为team id，_agent_i_obs[1]中记录的全为unique id，以此类推
            # 所以这里需要额外的team id为0 来标记这个格子为空
            hp = self.agent_health[agent_i]

            # If agent is alive
            # _agent_i_obs = self._one_hot_encoding(agent_i, self.n_agents)
            # _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            # _agent_i_obs += [self.agent_health[agent_i]]
            # _agent_i_obs += [1 if self._agent_cool else 0]  # flag if agent is cooling down
            # 首先判断hp是否大于0，若大于0，则获取agent_i的位置pos，然后进入循环，以pos为中心，
            # 取5x5的范围，并判断每个位置是否有效，若有效，则判断该位置是否有物体，若有物体，则判断物体类型，
            # 若是agent，则_type = 1，若是opp，则_type = -1，然后根据_type确定_id，
            # 并根据_id获取agent或opp的健康值，cool值，位置等信息，最后将_agent_i_obs的各个值拼接成_obs，最后返回_obs。
            if hp > 0:
                pos = self.agent_pos[agent_i]
                for row in range(0, 5):
                    for col in range(0, 5):
                        if self.is_valid([row + (pos[0] - 2), col + (pos[1] - 2)]) and (
                                PRE_IDS['empty'] not in self._full_obs[row + (pos[0] - 2)][col + (pos[1] - 2)]):
                            x = self._full_obs[row + pos[0] - 2][col + pos[1] - 2]
                            # x应该是一张二维列表，记录了当前格子的状态和id
                            if PRE_IDS['agent'] in x:
                                _type = 1
                            if PRE_IDS['opponent'] in x:
                                _type = -1
                            if PRE_IDS['neutral'] in x:
                                _type = 2
                            # _type = 1 if PRE_IDS['agent'] in x else -1
                            _id = int(x[1:]) - 1  # id
                            _agent_i_obs[0][row][col] = _type
                            _agent_i_obs[1][row][col] = _id
                            if _type == 1:
                                _agent_i_obs[2][row][col] = self.agent_health[_id]
                                _agent_i_obs[3][row][col] = self._agent_cool[_id]
                                entity_position = self.agent_pos[_id]
                            if _type == -1:
                                _agent_i_obs[2][row][col] = self.opp_health[_id]
                                _agent_i_obs[3][row][col] = self._opp_cool[_id]
                                entity_position = self.opp_pos[_id]
                            if _type == 2:
                                _agent_i_obs[2][row][col] = self.neu_health[_id]
                                _agent_i_obs[3][row][col] = self._neu_cool[_id]
                                entity_position = self.neu_pos[_id]
                            _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1
                            _agent_i_obs[4][row][col] = entity_position[0] / self._grid_shape[0]  # x-coordinate
                            _agent_i_obs[5][row][col] = entity_position[1] / self._grid_shape[1]  # y-coordinate
                            # _agent_i_obs[2][row][col] = self.agent_health[_id] if _type == 1 else self.opp_health[_id]
                            # _agent_i_obs[3][row][col] = self._agent_cool[_id] if _type == 1 else self._opp_cool[_id]
                            # _agent_i_obs[3][row][col] = 1 if _agent_i_obs[3][row][col] else -1  # cool/uncool
                            # entity_position = self.agent_pos[_id] if _type == 1 else self.opp_pos[_id]
                            # _agent_i_obs[4][row][col] = entity_position[0] / self._grid_shape[0]  # x-coordinate
                            # _agent_i_obs[5][row][col] = entity_position[1] / self._grid_shape[1]  # y-coordinate

            _agent_i_obs = _agent_i_obs.flatten().tolist()
            _obs.append(_agent_i_obs)
        return _obs

    def get_state(self):  # 这里也需要修改
        state = np.zeros((self.n_agents + self._n_opponents + self.n_neutrals, 6))
        # agent info
        for agent_i in range(self.n_agents):
            hp = self.agent_health[agent_i]
            if hp > 0:
                pos = self.agent_pos[agent_i]
                feature = np.array([1, agent_i, hp, 1 if self._agent_cool[agent_i] else -1,
                                    pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]], dtype=np.float)
                state[agent_i] = feature
        # feature存储的分别是类型（1代表自己队伍，编号，血量，攻击冷却状态，以及xy的比例坐标值）
        # opponent info
        for opp_i in range(self._n_opponents):
            opp_hp = self.opp_health[opp_i]
            if opp_hp > 0:
                pos = self.opp_pos[opp_i]
                feature = np.array([-1, opp_i, opp_hp, 1 if self._opp_cool[opp_i] else -1,
                                    pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]], dtype=np.float)
                state[opp_i + self.n_agents] = feature

        # neutral info
        for neu_i in range(self._n_opponents):
            neu_hp = self.neu_health[neu_i]
            if neu_hp > 0:
                pos = self.neu_pos[neu_i]
                feature = np.array([2, neu_i, neu_hp, 1 if self._neu_cool[neu_i] else -1,
                                    pos[0] / self._grid_shape[0], pos[1] / self._grid_shape[1]], dtype=np.float)
                state[neu_i + self._n_opponents + self.n_agents] = feature
        return state.flatten()

    def get_state_size(self):
        return (self.n_neutrals + self.n_agents + self._n_opponents) * 6

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    # 创建初始网格

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_opp_view(self, opp_i):
        self._full_obs[self.opp_prev_pos[opp_i][0]][self.opp_prev_pos[opp_i][1]] = PRE_IDS['empty']
        self._full_obs[self.opp_pos[opp_i][0]][self.opp_pos[opp_i][1]] = PRE_IDS['opponent'] + str(opp_i + 1)

    def __update_neu_view(self, neu_i):
        self._full_obs[self.neu_prev_pos[neu_i][0]][self.neu_prev_pos[neu_i][1]] = PRE_IDS['empty']
        self._full_obs[self.neu_pos[neu_i][0]][self.neu_pos[neu_i][1]] = PRE_IDS['neutral'] + str(neu_i + 1)

    # 更新各个智能体的视野
    def __init_full_obs(self):
        """ Each team consists of m = 5 agents and their initial positions are sampled uniformly in a 5 × 5
        square around the team center, which is picked uniformly in the grid.每个团队由m = 5个智能体组成，
        其初始位置在团队中心周围5 × 5的正方形内均匀采样，并在网格内均匀选取
        """
        self._full_obs = self.__create_grid()

        # select agent team center
        # Note : Leaving space from edges so as to have a 5x5 grid around it
        agent_team_center = self.np_random.randint(2, self._grid_shape[0] - 3), self.np_random.randint(2,
                                                                                                       self._grid_shape[
                                                                                                           1] - 3)
        # agent_team_center = [2, 6]
        # position = [[1, 6], [2, 5], [2, 6], [2, 7], [3, 6]]
        # for agent_i in range(self.n_agents):
        #     pos = position[agent_i]
        #     if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
        #         self.agent_prev_pos[agent_i] = pos
        #         self.agent_pos[agent_i] = pos
        #         self.__update_agent_view(agent_i)

        # randomly select agent pos
        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(agent_team_center[0] - 2, agent_team_center[0] + 2),
                       self.np_random.randint(agent_team_center[1] - 2, agent_team_center[1] + 2)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.agent_prev_pos[agent_i] = pos
                    self.agent_pos[agent_i] = pos
                    self.__update_agent_view(agent_i)
                    break

        # select opponent team center
        while True:
            pos = self.np_random.randint(2, self._grid_shape[0] - 3), self.np_random.randint(2, self._grid_shape[1] - 3)
            # pos = [8, 2]
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                opp_team_center = pos
                break
        # position = [[7, 2], [8, 1], [8, 2], [8, 3], [9, 2]]
        # for opp_i in range(self._n_opponents):
        #     pos = position[opp_i]
        #     if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
        #         self.opp_prev_pos[opp_i] = pos
        #         self.opp_pos[opp_i] = pos
        #         self.__update_opp_view(opp_i)
        # randomly select opponent pos
        for opp_i in range(self._n_opponents):
            while True:
                pos = [self.np_random.randint(opp_team_center[0] - 2, opp_team_center[0] + 2),
                       self.np_random.randint(opp_team_center[1] - 2, opp_team_center[1] + 2)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.opp_prev_pos[opp_i] = pos
                    self.opp_pos[opp_i] = pos
                    self.__update_opp_view(opp_i)
                    break

        # select neutral team center
        while True:
            pos = self.np_random.randint(2, self._grid_shape[0] - 3), self.np_random.randint(2, self._grid_shape[
                1] - 3)
            # pos = [7, 11]
            if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                neu_team_center = pos
                break
        # position = [[6, 11], [, 1], [8, 2], [8, 3], [9, 2]]
        # randomly select neutral pos
        for neu_i in range(self.n_neutrals):
            while True:
                pos = [self.np_random.randint(neu_team_center[0] - 2, neu_team_center[0] + 2),
                       self.np_random.randint(neu_team_center[1] - 2, neu_team_center[1] + 2)]
                if self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty']:
                    self.neu_prev_pos[neu_i] = pos
                    self.neu_pos[neu_i] = pos
                    self.__update_neu_view(neu_i)
                    break

        self.__draw_base_img()

    def reset(self):
        self._step_count = 0
        self._steps_beyond_done = None
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_health = {_: self._init_health for _ in range(self.n_agents)}
        self.opp_health = {_: self._init_health for _ in range(self._n_opponents)}
        self.neu_health = {_: self._init_health for _ in range(self.n_neutrals)}
        self._agent_cool = {_: True for _ in range(self.n_agents)}
        self._agent_cool_step = {_: 0 for _ in range(self.n_agents)}
        self._opp_cool = {_: True for _ in range(self._n_opponents)}
        self._opp_cool_step = {_: 0 for _ in range(self._n_opponents)}
        self._neu_cool = {_: True for _ in range(self.n_neutrals)}
        self._neu_cool_step = {_: 0 for _ in range(self.n_neutrals)}
        self._agent_dones = [False for _ in range(self.n_agents)]

        self.__init_full_obs()

        # For debug only
        self._agents_trace = {_: [self.agent_pos[_]] for _ in range(self.n_agents)}
        self._opponents_trace = {_: [self.opp_pos[_]] for _ in range(self._n_opponents)}
        self._neutrals_trace = {_: [self.neu_pos[_]] for _ in range(self.n_neutrals)}

        return self.get_agent_obs()

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        img = copy.copy(self._base_img)

        # draw agents
        for agent_i in range(self.n_agents):
            if self.agent_health[agent_i] > 0:
                fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
                write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        # draw opponents
        for opp_i in range(self._n_opponents):
            if self.opp_health[opp_i] > 0:
                fill_cell(img, self.opp_pos[opp_i], cell_size=CELL_SIZE, fill=OPPONENT_COLOR)
                write_cell_text(img, text=str(opp_i + 1), pos=self.opp_pos[opp_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        # draw neutrals
        for neu_i in range(self.n_neutrals):
            if self.neu_health[neu_i] > 0:
                fill_cell(img, self.neu_pos[neu_i], cell_size=CELL_SIZE, fill=NEUTRAL_COLOR)
                write_cell_text(img, text=str(neu_i + 1), pos=self.neu_pos[neu_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.3)

        img = np.asarray(img)
        # 如果`mode`是`rgb_array`，那么直接返回原始的图像数据`img`。
        #
        # 如果`mode`是`human`，
        # 则会引入`gym.envs.classic_control.rendering`模块，并检查`self.viewer`是否已经初始化。
        # 如果没有，就创建一个`SimpleImageViewer`对象并赋值给`self.viewer`。
        #
        # 接下来，使用`self.viewer.imshow(img)`将图像显示出来，
        # 并返回一个布尔值`self.viewer.isopen`，表示窗口是否是开启状态。
        # 此时，可以通过判断`self.viewer.isopen`来终止程序或采取其他的操作。
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self.agent_prev_pos[agent_i] = curr_pos
            self.__update_agent_view(agent_i)
            self._agents_trace[agent_i].append(next_pos)

    def __update_opp_pos(self, opp_i, move):

        curr_pos = copy.copy(self.opp_pos[opp_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.opp_pos[opp_i] = next_pos
            self.opp_prev_pos[opp_i] = curr_pos
            self.__update_opp_view(opp_i)
            self._opponents_trace[opp_i].append(next_pos)

    # 添加neu的__update_neu_pos
    def __update_neu_pos(self, neu_i, move):

        curr_pos = copy.copy(self.neu_pos[neu_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.neu_pos[neu_i] = next_pos
            self.neu_prev_pos[neu_i] = curr_pos
            self.__update_neu_view(neu_i)
            self._neutrals_trace[neu_i].append(next_pos)

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    # 判断是否出界以及是否为空
    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    # 目标在不在观察范围内
    @staticmethod
    def is_visible(source_pos, target_pos):
        """
        Checks if the target_pos is in the visible range(5x5)  of the source pos

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return (source_pos[0] - 2) <= target_pos[0] <= (source_pos[0] + 2) \
               and (source_pos[1] - 2) <= target_pos[1] <= (source_pos[1] + 2)

    # 目标在不在攻击范围内以及自己是否可以攻击
    @staticmethod
    def is_fireable(source_cooling_down, source_pos, target_pos):
        """
        Checks if the target_pos is in the firing range(5x5)

        :param source_pos: Coordinates of the source
        :param target_pos: Coordinates of the target
        :return:
        """
        return source_cooling_down and (source_pos[0] - 1) <= target_pos[0] <= (source_pos[0] + 1) \
               and (source_pos[1] - 1) <= target_pos[1] <= (source_pos[1] + 1)

    def reduce_distance_move(self, opp_i, source_pos, agent_i, target_pos):
        # Todo: makes moves Enum
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append('UP')
        elif source_pos[0] < target_pos[0]:
            _moves.append('DOWN')

        if source_pos[1] > target_pos[1]:
            _moves.append('LEFT')
        elif source_pos[1] < target_pos[1]:
            _moves.append('RIGHT')
        # 首先，它检查_moves变量是否为空，如果为空，则打印游戏中agent和opponent的相关信息和当前位置，
        # 然后抛出一个错误，表示当前位置存在两个实体。
        # 否则，选择_moves中的一个动作，并将其转换为ACTION_MEANING中的动作，然后返回转换后的动作。
        if len(_moves) == 0:
            print(self._step_count, source_pos, target_pos)
            print("agent-{}, hp={}, move_trace={}".format(agent_i, self.agent_health[agent_i],
                                                          self._agents_trace[agent_i]))
            print(
                "opponent-{}, hp={}, move_trace={}".format(opp_i, self.opp_health[opp_i], self._opponents_trace[opp_i]))
            raise AssertionError("One place exists 2 entities!")
        move = self.np_random.choice(_moves)
        for k, v in ACTION_MEANING.items():
            if move.lower() == v.lower():
                move = k
                break
        return move


    def align(self,target_type):
        # opp_all_health = sum([v for k, v in self.opp_health.items()])
        # if target_type == 2:
        #     if opp_all_health != 0:
        #         return True
        #     else:
        #         return False
        # else:
            return False

    # new
    def reduce_distance(self, source_type, source_i, source_pos, target_type, target_i, target_pos):
        # Todo: makes moves Enum
        _moves = []
        if source_pos[0] > target_pos[0]:
            _moves.append('UP')
        elif source_pos[0] < target_pos[0]:
            _moves.append('DOWN')

        if source_pos[1] > target_pos[1]:
            _moves.append('LEFT')
        elif source_pos[1] < target_pos[1]:
            _moves.append('RIGHT')
        # 首先，它检查_moves变量是否为空，如果为空，则打印游戏中agent和opponent的相关信息和当前位置，
        # 然后抛出一个错误，表示当前位置存在两个实体。
        # 否则，选择_moves中的一个动作，并将其转换为ACTION_MEANING中的动作，然后返回转换后的动作。
        if len(_moves) == 0:
            print(self._step_count, source_pos, target_pos)
            if source_type == 1 and target_type == -1:
                print("agent-{}, hp={}, move_trace={}".format(source_i, self.agent_health[source_i],
                                                              self._agents_trace[source_i]))
                print(
                    "opponent-{}, hp={}, move_trace={}".format(target_i, self.opp_health[target_i],
                                                               self._opponents_trace[target_i]))
            if source_type == 1 and target_type == 2:
                print("agent-{}, hp={}, move_trace={}".format(source_i, self.agent_health[source_i],
                                                              self._agents_trace[source_i]))
                print(
                    "neutral-{}, hp={}, move_trace={}".format(target_i, self.neu_health[target_i],
                                                              self._neutrals_trace[target_i]))
            if source_type == -1 and target_type == 1:
                print("opponent-{}, hp={}, move_trace={}".format(source_i, self.opp_health[source_i],
                                                                 self._opponents_trace[source_i]))
                print(
                    "agent-{}, hp={}, move_trace={}".format(target_i, self.agent_health[target_i],
                                                            self._agents_trace[target_i]))
            if source_type == -1 and target_type == 2:
                print("opponent-{}, hp={}, move_trace={}".format(source_i, self.opp_health[source_i],
                                                                 self._opponents_trace[source_i]))
                print(
                    "neutral-{}, hp={}, move_trace={}".format(target_i, self.neu_health[target_i],
                                                              self._neutrals_trace[target_i]))
            if source_type == 2 and target_type == 1:
                print("neutral-{}, hp={}, move_trace={}".format(source_i, self.neu_health[source_i],
                                                                self._neutrals_trace[source_i]))
                print(
                    "agent-{}, hp={}, move_trace={}".format(target_i, self.agent_health[target_i],
                                                            self._agents_trace[target_i]))
            if source_type == 2 and target_type == -1:
                print("neutral-{}, hp={}, move_trace={}".format(source_i, self.neu_health[source_i],
                                                                self._neutrals_trace[source_i]))
                print(
                    "opponent-{}, hp={}, move_trace={}".format(target_i, self.opp_health[target_i],
                                                               self._opponents_trace[target_i]))
            raise AssertionError("One place exists 2 entities!")
        move = self.np_random.choice(_moves)
        for k, v in ACTION_MEANING.items():
            if move.lower() == v.lower():
                move = k
                break
        return move

    # @property的作用是为了让这个函数能够像属性一样被self使用，opp_action = self.opps_action
    @property
    def opps_action(self):
        """
        Opponent bots follow a hardcoded policy.

        The bot policy is to attack the nearest enemy agent if it is within its firing range. If not,
        it approaches the nearest visible enemy agent within visual range. An agent is visible to all bots if it
        is inside the visual range of any individual bot. This shared vision gives an advantage to the bot team.

        :return:
        """

        visible_agents = set([])  # 这里的agents包括agent方和neutral方
        opp_agent_distance = {_: [] for _ in range(self._n_opponents)}
        # 遍历了所有的对手智能体和自己的智能体，对可见的智能体（未被其他对手智能体挡住）进行标记，
        # 在 opp_agent_distance 中记录其和每个对手智能体的曼哈顿距离。
        for opp_i, opp_pos in self.opp_pos.items():
            if not self.align(-1):
                for agent_i, agent_pos in self.agent_pos.items():
                    if agent_i not in visible_agents and self.agent_health[agent_i] > 0 \
                            and self.is_visible(opp_pos, agent_pos):
                        visible_agents.add(agent_i)
                    distance = abs(agent_pos[0] - opp_pos[0]) + abs(agent_pos[1] - opp_pos[1])  # manhattan distance曼哈顿距离
                    opp_agent_distance[opp_i].append([distance, agent_i])
            # 新增
            for neu_i, neu_pos in self.neu_pos.items():
                if (neu_i + 5) not in visible_agents and self.neu_health[neu_i] > 0 \
                        and self.is_visible(opp_pos, neu_pos):
                    visible_agents.add(neu_i + 5)
                distance = abs(neu_pos[0] - opp_pos[0]) + abs(neu_pos[1] - opp_pos[1])  # manhattan distance曼哈顿距离
                opp_agent_distance[opp_i].append([distance, neu_i + 5])
        # 通过对 opp_agent_distance 中距离进行排序，逐个判断可见智能体是否可以开火攻击或者靠近该智能体。
        # 如果找到了合适目标，则直接返回该目标的编号；
        # 如果所有的遍历都结束仍未找到符合要求的目标，则随机选择一个行动。
        # 如果对手智能体已经死亡，只能进行空操作。
        # 最后，将所有的对手智能体的行动决策放入 opp_action_n 中返回。
        opp_action_n = []
        for opp_i in range(self._n_opponents):
            action = None
            for _, agent_i in sorted(opp_agent_distance[opp_i]):
                if agent_i in visible_agents:
                    if agent_i >= 0 and agent_i < 5 and self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i],
                                                                         self.agent_pos[agent_i]):
                        action = agent_i + 5
                    elif agent_i > 4 and agent_i < 10 and self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i],
                                                                           self.neu_pos[agent_i - 5]):
                        action = agent_i + 5

                    elif self.opp_health[opp_i] > 0 and agent_i >= 0 and agent_i < 5:

                        action = self.reduce_distance(-1, opp_i, self.opp_pos[opp_i], 1, agent_i,
                                                      self.agent_pos[agent_i])
                    elif self.opp_health[opp_i] > 0 and agent_i > 4 and agent_i < 10:  # 要修改这边的话首先必须改使用reduce_distance
                        neu_i = agent_i - 5
                        action = self.reduce_distance(-1, opp_i, self.opp_pos[opp_i], 2, neu_i, self.neu_pos[neu_i])
                    break
            if action is None:
                if self.opp_health[opp_i] > 0:
                    # logger.debug('No visible agent for enemy:{}'.format(opp_i))
                    action = self.np_random.choice(range(5))
                else:
                    action = 4  # dead opponent could only execute 'no-op' action.
            opp_action_n.append(action)
        return opp_action_n

    @property
    def neus_action(self):  # 该函数中agent_i: 0-4代表agent，5-9代表opp ;action： 5-9代表攻击agent 10-14代表攻击opp
        visible_agents = set([])  # 这里的agents包括agent方和neutral方
        neu_agent_distance = {_: [] for _ in range(self.n_neutrals)}
        for neu_i, neu_pos in self.neu_pos.items():
            if not self.align(2):
                for agent_i, agent_pos in self.agent_pos.items():
                    if agent_i not in visible_agents and self.agent_health[agent_i] > 0 \
                            and self.is_visible(neu_pos, agent_pos):
                        visible_agents.add(agent_i)
                    distance = abs(agent_pos[0] - neu_pos[0]) + abs(agent_pos[1] - neu_pos[1])  # manhattan distance曼哈顿距离
                    neu_agent_distance[neu_i].append([distance, agent_i])

            for opp_i, opp_pos in self.opp_pos.items():
                if (opp_i + 5) not in visible_agents and self.opp_health[opp_i] > 0 \
                        and self.is_visible(neu_pos, opp_pos):
                    visible_agents.add(opp_i + 5)
                distance = abs(opp_pos[0] - neu_pos[0]) + abs(opp_pos[1] - neu_pos[1])  # manhattan distance曼哈顿距离
                neu_agent_distance[neu_i].append([distance, opp_i + 5])
        neu_action_n = []
        for neu_i in range(self.n_neutrals):
            action = None
            for _, agent_i in sorted(neu_agent_distance[neu_i]):
                if agent_i in visible_agents:
                    if agent_i >= 0 and agent_i < 5 and self.is_fireable(self._neu_cool[neu_i], self.neu_pos[neu_i],
                                                                         self.agent_pos[agent_i]):
                        action = agent_i + 5
                    elif agent_i > 4 and agent_i < 10 and self.is_fireable(self._neu_cool[neu_i], self.neu_pos[neu_i],
                                                                           self.opp_pos[agent_i - 5]):
                        action = agent_i + 5


                    elif self.neu_health[neu_i] > 0 and agent_i >= 0 and agent_i < 5:
                        action = self.reduce_distance(2, neu_i, self.neu_pos[neu_i], 1, agent_i,
                                                      self.agent_pos[agent_i])
                    elif self.neu_health[neu_i] > 0 and agent_i > 4 and agent_i < 10:
                        action = self.reduce_distance(2, neu_i, self.neu_pos[neu_i], -1, agent_i - 5,
                                                      self.opp_pos[agent_i - 5])
                    break
            if action is None:
                if self.neu_health[neu_i] > 0:
                    # logger.debug('No visible agent for enemy:{}'.format(opp_i))
                    action = self.np_random.choice(range(5))
                else:
                    action = 4  # dead opponent could only execute 'no-op' action.
            neu_action_n.append(action)
        return neu_action_n

    @property
    def agents_same_action(self):  # 该函数中agent_i: 0-4代表opp，5-9代表neu ;action： 5-9代表攻击opp 10-14代表攻击neu
        visible_agents = set([])  # 这里的agents包括opponent方和neutral方
        agent_agent_distance = {_: [] for _ in range(self.n_agents)}
        for agent_i, agent_pos in self.agent_pos.items():
            if not self.align(-1):
                for opp_i, opp_pos in self.opp_pos.items():
                    if opp_i not in visible_agents and self.opp_health[opp_i] > 0 \
                            and self.is_visible(agent_pos, opp_pos):
                        visible_agents.add(opp_i)
                    distance = abs(opp_pos[0] - agent_pos[0]) + abs(opp_pos[1] - agent_pos[1])  # manhattan distance曼哈顿距离
                    agent_agent_distance[agent_i].append([distance, opp_i])
            if not self.align(2):
                for neu_i, neu_pos in self.neu_pos.items():
                    if (neu_i + 5) not in visible_agents and self.neu_health[neu_i] > 0 \
                            and self.is_visible(agent_pos, neu_pos):
                        visible_agents.add(neu_i + 5)
                    distance = abs(neu_pos[0] - agent_pos[0]) + abs(neu_pos[1] - agent_pos[1])  # manhattan distance曼哈顿距离
                    agent_agent_distance[agent_i].append([distance, neu_i + 5])
        agent_action_n = []
        for myself_i in range(self.n_agents):
            action = None
            for _, agent_i in sorted(agent_agent_distance[myself_i]):
                if agent_i in visible_agents:
                    if agent_i >= 0 and agent_i < 5 and self.is_fireable(self._agent_cool[myself_i],
                                                                         self.agent_pos[myself_i],
                                                                         self.opp_pos[agent_i]):
                        action = agent_i + 5
                    elif agent_i > 4 and agent_i < 10 and self.is_fireable(self._agent_cool[myself_i],
                                                                           self.agent_pos[myself_i],
                                                                           self.neu_pos[agent_i - 5]):
                        action = agent_i + 5


                    elif self.agent_health[myself_i] > 0 and agent_i >= 0 and agent_i < 5:
                        action = self.reduce_distance(1, myself_i, self.agent_pos[myself_i], -1, agent_i,
                                                      self.opp_pos[agent_i])
                    elif self.agent_health[myself_i] > 0 and agent_i > 4 and agent_i < 10:
                        action = self.reduce_distance(1, myself_i, self.agent_pos[myself_i], 2, agent_i - 5,
                                                      self.neu_pos[agent_i - 5])
                    break
            if action is None:
                if self.agent_health[myself_i] > 0:
                    # logger.debug('No visible agent for enemy:{}'.format(opp_i))
                    action = self.np_random.choice(range(5))
                else:
                    action = 4  # dead opponent could only execute 'no-op' action.
            agent_action_n.append(action)
        return agent_action_n

    def step(self, agents_action):
        # agents_action = self.agents_same_action
        assert (self._step_count is not None), \
            "Call reset before using step method."

        assert len(agents_action) == self.n_agents

        self._step_count += 1
        # rewards貌似只针对agent
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # What's the confusion?
        # What if agents attack each other at the same time? Should both of them be effected?
        # Ans: I guess, yes
        # What if other agent moves before the attack is performed in the same time-step.
        # Ans: May be, I can process all the attack actions before move directions to ensure attacks have their effect.

        # processing attacks
        agent_health, opp_health, neu_health = copy.copy(self.agent_health), copy.copy(self.opp_health), copy.copy(
            self.neu_health)
        for agent_i, action in enumerate(agents_action):  # 同时获取索引和值
            if self.agent_health[agent_i] > 0:
                # 新增
                if action > 4 + self._n_opponents:  # 攻击neutrals
                    target_neu = action - 10
                    if self.is_fireable(self._agent_cool[agent_i], self.agent_pos[agent_i], self.neu_pos[target_neu]) \
                            and neu_health[target_neu] > 0:
                        # Fire
                        neu_health[target_neu] -= 1  # 掉一滴血
                        rewards[agent_i] += 2  # 获得2点奖励

                        # Update agent cooling down
                        self._agent_cool[agent_i] = False  # 进入攻击冷却
                        self._agent_cool_step[agent_i] = self._step_cool  # 更新下一次攻击需要的时长

                        # Remove neu from the map
                        if neu_health[target_neu] == 0:
                            pos = self.neu_pos[target_neu]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']  # 敌方死亡移出地图
                if action > 4 and action < 5 + self._n_opponents:  # attack opponent
                    target_opp = action - 5
                    if self.is_fireable(self._agent_cool[agent_i], self.agent_pos[agent_i], self.opp_pos[target_opp]) \
                            and opp_health[target_opp] > 0:
                        # Fire
                        opp_health[target_opp] -= 1  # 掉一滴血
                        rewards[agent_i] += 2 # 获得2点奖励

                        # Update agent cooling down
                        self._agent_cool[agent_i] = False  # 进入攻击冷却
                        self._agent_cool_step[agent_i] = self._step_cool  # 更新下一次攻击需要的时长

                        # Remove opp from the map
                        if opp_health[target_opp] == 0:
                            pos = self.opp_pos[target_opp]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']  # 敌方死亡移出地图
                # 对每个代理的冷却时间进行更新。
                # 如果该代理攻击行动的冷却时间已经达到或超过0，则将其冷却状态设为 已冷却
                # Update agent cooling down
                self._agent_cool_step[agent_i] = max(self._agent_cool_step[agent_i] - 1, 0)
                if self._agent_cool_step[agent_i] == 0 and not self._agent_cool[agent_i]:
                    self._agent_cool[agent_i] = True

        opp_action = self.opps_action
        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                # target_agent = action - 5
                if action > 9:  # 攻击neutrals
                    target_neu = action - 10
                    if self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i], self.neu_pos[target_neu]) \
                            and neu_health[target_neu] > 0:
                        # Fire
                        neu_health[target_neu] -= 1  # 掉一滴血
                        # rewards[opp_i] += 1  # 获得一点奖励

                        # Update opp cooling down
                        self._opp_cool[opp_i] = False  # 进入攻击冷却
                        self._opp_cool_step[opp_i] = self._step_cool  # 更新下一次攻击需要的时长

                        # Remove neu from the map
                        if neu_health[target_neu] == 0:
                            pos = self.neu_pos[target_neu]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']  # 敌方死亡移出地图
                if action > 4 and action < 10:  # attack actions >4为攻击agent，反之为移动
                    target_agent = action - 5
                    if self.is_fireable(self._opp_cool[opp_i], self.opp_pos[opp_i], self.agent_pos[target_agent]) \
                            and agent_health[target_agent] > 0:
                        # Fire
                        agent_health[target_agent] -= 1
                        rewards[target_agent] -= 1

                        # Update opp cooling down
                        self._opp_cool[opp_i] = False
                        self._opp_cool_step[opp_i] = self._step_cool

                        # Remove agent from the map
                        if agent_health[target_agent] == 0:
                            pos = self.agent_pos[target_agent]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']
                # Update opp cooling down
                self._opp_cool_step[opp_i] = max(self._opp_cool_step[opp_i] - 1, 0)
                if self._opp_cool_step[opp_i] == 0 and not self._opp_cool[opp_i]:
                    self._opp_cool[opp_i] = True
        neu_action = self.neus_action
        for neu_i, action in enumerate(neu_action):
            if self.neu_health[neu_i] > 0:
                # target_agent = action - 5
                if action > 9:  # 攻击opponents
                    target_opp = action - 10
                    if self.is_fireable(self._neu_cool[neu_i], self.neu_pos[neu_i], self.opp_pos[target_opp]) \
                            and opp_health[target_opp] > 0:
                        # Fire
                        opp_health[target_opp] -= 1  # 掉一滴血
                        # rewards[neu_i] += 1  # 获得一点奖励

                        # Update neu cooling down
                        self._neu_cool[neu_i] = False  # 进入攻击冷却
                        self._neu_cool_step[neu_i] = self._step_cool  # 更新下一次攻击需要的时长

                        # Remove opp from the map
                        if opp_health[target_opp] == 0:
                            pos = self.opp_pos[target_opp]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']  # 敌方死亡移出地图
                if action > 4 and action < 10:  # attack agents
                    target_agent = action - 5
                    if self.is_fireable(self._neu_cool[neu_i], self.neu_pos[neu_i], self.agent_pos[target_agent]) \
                            and agent_health[target_agent] > 0:
                        # Fire
                        agent_health[target_agent] -= 1
                        rewards[target_agent] -= 1

                        # Update neu cooling down
                        self._neu_cool[neu_i] = False
                        self._neu_cool_step[neu_i] = self._step_cool

                        # Remove agent from the map
                        if agent_health[target_agent] == 0:
                            pos = self.agent_pos[target_agent]
                            self._full_obs[pos[0]][pos[1]] = PRE_IDS['empty']
                # Update neu cooling down
                self._neu_cool_step[neu_i] = max(self._neu_cool_step[neu_i] - 1, 0)
                if self._neu_cool_step[neu_i] == 0 and not self._neu_cool[neu_i]:
                    self._neu_cool[neu_i] = True

        self.agent_health, self.opp_health, self.neu_health = agent_health, opp_health, neu_health

        # process move actions
        # 对于每一个智能体，判断该智能体是否存活，
        # 并判断其行动是否为合法的运动行为，如果是，则将其更新到新的位置。
        for agent_i, action in enumerate(agents_action):
            if self.agent_health[agent_i] > 0:
                if action <= 4:
                    self.__update_agent_pos(agent_i, action)

        for opp_i, action in enumerate(opp_action):
            if self.opp_health[opp_i] > 0:
                if action <= 4:
                    self.__update_opp_pos(opp_i, action)
        # 新增
        for neu_i, action in enumerate(neu_action):
            if self.neu_health[neu_i] > 0:
                if action <= 4:
                    self.__update_neu_pos(neu_i, action)
        # step overflow or all opponents dead
        # 当环境中步数达到最大限制，
        # 或者所有的对手都死亡，或者所有的智能体都死亡时，将标记各智能体的状态。

        agent_all_health = sum([v for k, v in self.agent_health.items()])
        opp_all_health = sum([v for k, v in self.opp_health.items()])
        neu_all_health = sum([v for k, v in self.neu_health.items()])
        if (self._step_count >= self._max_steps) \
                or (opp_all_health + neu_all_health == 0) \
                or (agent_all_health + neu_all_health == 0) \
                or (agent_all_health + opp_all_health == 0):
            self._agent_dones = [True for _ in range(self.n_agents)]

        # 对于每一个智能体，将其获得的回报值与总回报值相加
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # 如果所有智能体都完成了任务，则检查游戏是否已经结束。
        # 如果结束了，则将结束标记作为结果返回。
        # 如果未结束，则提示已经完成游戏，但是还是调用了step()方法，并记录次数。
        # Check for episode overflow
        if all(self._agent_dones):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive "
                        "'done = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1


        return self.get_agent_obs(), rewards, self._agent_dones, {'health': self.agent_health, 'win':True if (opp_all_health + neu_all_health == 0) else False }

    # 返回一个随机种子值的列表，seed的作用就是保证初始化的随机值是一样的
    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 15

WALL_COLOR = 'black'
AGENT_COLOR = 'red'
NEUTRAL_COLOR = 'green'
OPPONENT_COLOR = 'blue'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'wall': 'W',
    'empty': '0',
    'agent': 'A',
    'opponent': 'X',
    'neutral': 'N',
}

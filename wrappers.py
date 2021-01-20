from collections import deque
from copy import copy
import gym.spaces as spaces
import gym
import torch as th
import numpy as np
import pandas as pd
TD = 'trade_data'
TI = 'tech_indicators'
PV = 'private_vars'


def un_flatten(combined, trade_data_n, private_vars_n):

    trade_data = combined[:trade_data_n]
    tech_indicators = combined[trade_data_n: -private_vars_n]
    private_vars = combined[-private_vars_n:]
    obs = dict(trade_data=trade_data,
               tech_indicators=tech_indicators,
               private_vars=private_vars)
    return obs


def flatten_np(observation):
    obs = observation['trade_data'].tolist() + \
          observation['tech_indicators'] + \
          list(observation["private_vars"].values())
    combined = np.array(obs, dtype=np.float32)
    return combined


class GaussianCorruptObservation(gym.ObservationWrapper):
    """
    Observation wrapper to add gaussian noise to data
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env
        self.state = self.observation_space.sample()

    def observation(self, observation):
        combined = flatten_np(observation)
        combined += np.random.normal(size=combined.shape)
        self.state = combined
        return un_flatten(combined, len(observation[TD]), len(observation[PV]) )


class KLineWrapper(gym.ObservationWrapper):
    """
    Observation wrapper to add sdea noise to data
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env
        self.state = self.observation_space.sample()

    def observation(self, observation):
        # OHCLV
        assert isinstance(observation, dict), 'kline wrapper need unflattened dict'
        trade_data = observation['trade_data']
        u_s = abs(trade_data['high'] - trade_data['close'])
        l_s = abs(trade_data['open'] - trade_data['low'])
        b = abs(trade_data['close'] - trade_data['open'])
        c = b > 0

        obs = copy(observation)
        obs['trade_data'] = [u_s, l_s, b, c]
        return obs


class NormWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space

    def observation(self, observation):
        combined = observation['trade_data'].to_numpy()
        ti = np.array(observation['tech_indicators'], dtype=np.float32)
        combined = np.append(combined, (ti - np.min(ti)) / (np.max(ti) - np.min(ti)))
        pv = np.array(list(observation["private_vars"].values()), dtype=np.float32)
        combined = np.append(combined, pv)
        combined = (combined - np.min(combined))/(np.max(combined) - np.min(combined))

        return un_flatten(combined, len(observation[TD]), len(observation[PV]))


class StackWrapper(gym.ObservationWrapper):
    """
    Observation wrapper to add sdea noise to data
    """

    def __init__(self, env, n_stacks=28):
        super().__init__(env)
        self.env = env
        s = env.observation_space.sample()
        td_shape, ti_shape, pv_shape = (n_stacks, len(s[TD])),\
                                       (n_stacks, len(s[TI])),\
                                       (n_stacks, len(s[PV]))
        init = [env.observation_space.sample() for _ in range(n_stacks)]
        self.observation_space = gym.spaces.Dict(trade_data=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                           shape=td_shape),
                                                 tech_indicators=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                                shape=ti_shape),
                                                 private_vars=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                             shape=pv_shape))

        self.stock_index_stack = deque(init, maxlen=n_stacks)
        self.n_stacks = n_stacks
        self.state = self.observation_space.sample()

    def observation(self, observation):
        ob = copy(observation)
        ob['trade_data'] = list(observation[TD])
        ob['private_vars'] = list(observation[PV])

        self.stock_index_stack.append(ob)
        td = [d['trade_data'] for d in self.stock_index_stack]
        ti = [d['tech_indicators'] for d in self.stock_index_stack]
        pv = [d['private_vars'] for d in self.stock_index_stack]
        obs = dict(trade_data=td, tech_indicators=ti, private_vars=pv)
        self.state = obs
        return obs


class NumpyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.state = self.observation_space.sample()
        self.observation_space = env.observation_space

    def observation(self, observation):

        td = observation['trade_data'].to_numpy()
        ti = np.array(observation['tech_indicators'], dtype=np.float32)
        pv = np.array(list(observation["private_vars"].values()), dtype=np.float32)
        self.state = dict(trade_data=td, tech_indicators=ti, private_vars=pv)
        return self.state


class Turbulence(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space

    def observation(self, observation):
        index = self.env.index
        td = observation['trade_data'].to_numpy()
        ti = np.array(observation['tech_indicators'], dtype=np.float32)
        pv = np.array(list(observation["private_vars"].values()), dtype=np.float32)

        return dict(trade_data=td, tech_indicators=ti, private_vars=pv)
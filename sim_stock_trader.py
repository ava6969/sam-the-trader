import glob
import math
import random
from datetime import time
import colorama
import pytz
import tabulate
import datetime
from colorama import Fore
from scipy.signal import argrelmin, argrelmax

from config import tech_inds
from gym.utils import seeding

from downloader import add_tech_indicators, load_dataset

import gym
import numpy as np

NY = 'America/New_York'
import pandas as pd
import logging


class SimStockTrader(gym.Env):
    def __init__(self, tickers, resolution, initial_amount, reward_type,
                 max_shares, discrete_action, bins, window,
                 test, tech_indicators, filter_date, random_start,
                 start_date, end_date, df_override=None, worker_index=0,
                 episode_range=None,vector_index=0, log_every=15000):

        colorama.init()
        if episode_range:
            assert len(episode_range) > 0
            self.episode_range = episode_range
        else:
            self.episode_range = (15000, 150000)

        self.log_ctr   = 0
        self.w_idx = worker_index
        self.v_idx = vector_index
        self.random_select = tickers[:4] == 'RLOL'
        if self.random_select:
            self.tick = tickers
        else:
            assert vector_index < len(tickers)
            self.tick = tickers[vector_index]

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        ch  = logging.StreamHandler()
        ch.setLevel(level=logging.INFO)
        self.logger.addHandler(ch)
        self.random_start = random_start
        self.transaction_cost_pct = 0
        self.index = 0
        self.date_memory = []
        self.test = test
        self.tech_indicators = tech_indicators
        self.private_variables = 3
        self.trade_data = 5  # OHCLV
        self.date = None
        self.resolution = resolution
        self.reward_type = reward_type
        self.prev_action = 0
        self.state = None
        self.reward = None
        self.index = 0
        self.tic_df = None
        self.max_shares = max_shares
        self.asset_memory = []
        self.action_memory = []
        self.close_memory = []
        self.window = window
        self.log_every = log_every
        self.day_p = 0
        if df_override is None:
            self.train_data = self.clean_up(start_date, end_date, filter_date, test)
        else:
            self.train_data = df_override

        self.n_tech_ind = len(self.train_data.columns) - self.trade_data
        self.observation_space = gym.spaces.Dict(trade_data=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                           shape=(self.trade_data,)),
                                                 tech_indicators=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                                shape=(self.n_tech_ind,)),
                                                 private_vars=gym.spaces.Box(low=-np.inf, high=np.inf,
                                                                             shape=(self.private_variables,)))
        self.df = None
        assert window < len(self.train_data), f'{tickers}: window size{window} cant be greater than data frame' \
                                              f' length{len(self.train_data)}'

        # account stuffs
        self.initial_amount = initial_amount
        self.close_price = 0
        self.action=0
        self.balance = initial_amount
        self.stock_owned = 0
        self.discrete_action = discrete_action
        self.calc_qty = 0
        self.cost = 0
        self.trades = 0
        self.prev_close_price = 0
        self.prev_a_t= 0
        self.prev_b_t = 0
        self.prev_ddt = 0
        self.ddt_prev_a_t = 0

        self.day_init = 0.0
        self.day_profits = []

        if discrete_action:
            assert bins is not None, 'bins has to be set if using discrete actions'
            assert bins % 2 == 1, 'bin size has to be odd to account for 0'
            self.bins = bins
            self.action_arg = np.linspace(-max_shares, max_shares, bins)
            self.action_space = gym.spaces.Discrete(bins)
        else:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def clean_up(self, start_date, end_date, filter_date, test):
        # swap tech indicators
        if self.tech_indicators in tech_inds.keys():
            self.tech_indicators = tech_inds[self.tech_indicators]

        if self.random_select:
            concat_n = int(self.tick.split('-')[-1])
            tickers = glob.glob(f'datasets/{self.resolution}/*')
            tickers = np.random.choice(tickers, concat_n)
            tickers = [t.split('/')[-1].split('_')[0] for t in tickers]
            df_dict = list(load_dataset(tickers, self.tech_indicators, self.resolution).values())
            train_data = pd.concat(df_dict)
            self.tic_df = train_data['tic']
        else:
            df_dict = load_dataset(self.tick, self.tech_indicators, self.resolution)
            train_data = df_dict[self.tick]
        train_data.drop('tic', axis=1, inplace=True)
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(NY))
        start_date = max(train_data.index[0], start_date)
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').astimezone(pytz.timezone(NY))
        if start_date > end_date:
            end_date = train_data.index[-1]
        train_data = train_data[(train_data.index >= start_date) & (train_data.index < end_date)]

        if filter_date:
            logging.info('Filtering dates')
            train_data = train_data[
                ((train_data.index.time >= time(hour=9, minute=30, second=0, tzinfo=pytz.timezone(NY))) &
                 (train_data.index.time <= time(hour=11, minute=30, second=0, tzinfo=pytz.timezone(NY)))) |
                ((train_data.index.time >= time(hour=13, tzinfo=pytz.timezone(NY))) &
                 (train_data.index.time <= time(hour=15, tzinfo=pytz.timezone(NY))))]

        return train_data

    def update_state(self):
        data = self.df.iloc[self.index]
        self.date = self.df.index[self.index]
        self.date_memory.append(self.date)
        self.action_memory.append(self.action)
        self.close_price = data.close
        self.close_memory.append(self.close_price)
        sharpe = self.sharpe_ratio()
        private = {'balance': self.balance,
                   'sharpe': sharpe,
                   'stocks_owned': self.stock_owned}
        # state = data.tolist()
        self.state = dict(trade_data=data[:self.trade_data],
                          tech_indicators=data[self.trade_data:].tolist(),
                          private_vars=private)

    def reset(self):
        self.day_init = self.initial_amount
        self.day_profits = []
        self.balance = self.initial_amount
        self.stock_owned = 0
        self.asset_memory = [self.initial_amount]
        self.prev_a_t= 0
        self.prev_b_t = 0
        self.prev_ddt = 0
        self.ddt_prev_a_t = 0
        end = len(self.train_data) - self.episode_range[1]
        if end <= 0:
            end = int(0.75 * len(self.train_data))
        start =  np.random.randint(0, end ) if self.random_start else 0
        if self.random_start:
            end = min(len(self.train_data), start + np.random.randint(low=int(self.episode_range[0]),
                                                                      high=int(self.episode_range[1])))
            self.df = self.train_data.iloc[start:end, :]
        else:
            self.df = self.train_data.iloc[start:, :]
        self.index = 0
        if self.random_select:
            self.tick = self.tic_df[start]

        self.logger.info(f'{self.tick}[{self.w_idx}, {self.v_idx}] :- '
                         f'Start Date {self.df.index[self.index].to_pydatetime()} - End Date {self.df.index[-1].to_pydatetime()}')
        self.update_state()
        return self.state

    def terminal_condition(self):
        return self.index >= len(self.df) - 1

    def sell_stock(self, sell_qty):
        self.calc_qty = 0
        if self.stock_owned > 0:
            calc_sell_qty = int(min(abs(sell_qty), self.stock_owned))
            self.balance += self.close_price * calc_sell_qty * (1 - self.transaction_cost_pct)

            self.stock_owned -= calc_sell_qty
            self.cost += self.close_price * calc_sell_qty * self.transaction_cost_pct
            self.trades += 1
            self.calc_qty = calc_sell_qty

    def buy_stock(self, buy_qty):
        max_shares = self.balance // self.close_price
        calc_buy_qty = int(min(max_shares, buy_qty))

        self.balance -= self.close_price * calc_buy_qty * (1 + self.transaction_cost_pct)
        self.stock_owned += calc_buy_qty

        self.cost += self.close_price * calc_buy_qty * self.transaction_cost_pct
        self.trades += 1
        self.calc_qty = calc_buy_qty

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = self.terminal_condition()
        if done:
            msg = self.stat()
            self.logger.info(msg)
        else:
            self.action = self.scale_action(action)
            self.transact(self.action)
            self.prev_close_price = self.close_price

            # step in market data
            self.log_ctr += 1
            self.index += 1
            self.update_state()
            self.asset_memory.append(self.close_price * self.stock_owned + self.balance)
            self.reward = self.compute_reward()
            self.day_p += 1
            prev_date = self.date_memory[-2]
            if self.date.day != prev_date.day:
                daily_profit = self.asset_memory[-2] - self.day_init
                self.day_init = self.asset_memory[-1]
                self.day_profits.append(daily_profit)
                self.day_p = 0

            if self.log_ctr == self.log_every:
                print(flush=True)
                msg = self.stat()
                self.logger.info(msg)
                self.log_ctr = 0

        return self.state, self.reward, done, {}

    def live_update(self, df, balance, stocks_owned):
        self.df = add_tech_indicators(df, self.tech_indicators)
        self.balance = balance
        self.stock_owned = stocks_owned
        self.index = -1

    def stat(self):
        debug = Fore.CYAN if self.test else Fore.WHITE
        debug += f'worker {self.w_idx} - {self.tick}\n'
        debug += f"date: {self.date}\n"
        debug += f"account balance: {self.balance}\n"
        debug += "begin total asset: {}\n".format(self.asset_memory[0])
        debug += "Portfolio Value: {}\n".format(self.asset_memory[-1])
        profit = self.asset_memory[-1] - self.initial_amount
        if profit > 0:
            debug += Fore.GREEN
        elif profit < 0:
            debug += Fore.RED
        else:
            debug += Fore.LIGHTWHITE_EX

        debug += "Profit/Loss: {}\n".format(profit)
        debug += "Avg Daily Profit: {}\n".format(np.mean(self.day_profits))
        debug += "total trades: {}\n".format(self.trades)

        # weekly trades and cost
        sharpe = self.sharpe_ratio()
        debug += "Sharpe Ratio: {}\n".format(sharpe)

        debug += Fore.GREEN + "trade state: \n"
        tab = tabulate.tabulate([['Tickers'] + [self.tick ],
                                 ['Close'] + [str(self.close_price)],
                                 ['Stocks'] + [str(self.stock_owned)]],
                                headers="firstrow",
                                floatfmt=".2f")
        debug += tab
        return debug

    def transact(self, action):

        if action < 0:
            self.sell_stock(action)
        elif action > 0:
            self.buy_stock(action)

    def scale_action(self, action):
        if self.discrete_action:
            return self.action_arg[action]
        else:
            return self.max_shares * action

    def compute_reward(self):

        if self.reward_type == 'ALL':
            del_p = self.close_price - self.prev_close_price
            pc = del_p * self.action
            sr = self.sharpe_ratio()
            profit = (self.asset_memory[-1] - self.initial_amount) / self.initial_amount
            tot = pc + sr + profit
            return tot

        if self.reward_type == 'PC':
            del_p = self.close_price - self.prev_close_price
            res = del_p * self.action
            return res

        if self.reward_type == 'VOL_CLOSE':
            close = self.df.close[self.index]
            open_ = self.df.open[self.index]

            del_n = self.df.volume[self.index] - self.df.volume[self.index-1]
            if self.action > 0:
                return del_n * (close - open_)
            elif self.action < 0:
                return del_n * (open_ - close)

            return self.stock_owned * close + self.balance - self.initial_amount

        elif self.reward_type == 'PC1':
            return self.asset_memory[-1] - self.asset_memory[-2]

        elif self.reward_type == 'PC_PERCENT':
            return (self.asset_memory[-1] - self.asset_memory[-2]) / self.asset_memory[-2]

        elif self.reward_type == 'PROFIT':
            return (self.asset_memory[-1] - self.initial_amount) / self.initial_amount

        elif self.reward_type == 'SR':
            return self.sharpe_ratio()

        elif self.reward_type == 'DSR':
            return self.diff_sharpe_ratio()

        elif self.reward_type == 'DDR':
            return self.diff_draw_down()

        elif self.reward_type == 'MULTI':
            return self.multi_obj()

    def diff_sharpe_ratio(self):
        del_p = self.close_price - self.prev_close_price
        r_t = del_p * self.action
        d_a = r_t - self.prev_a_t
        d_b = r_t ** 2 - self.prev_b_t
        new_a_t = self.prev_a_t + 0.01 * d_a
        new_b_t = self.prev_b_t + 0.01 * d_b

        denom = (self.prev_b_t - (self.prev_a_t ** 2)) ** 1.5
        denom = denom if denom else 1
        d_t = (self.prev_b_t * d_a - 0.5 * self.prev_a_t * d_b) / denom
        self.prev_a_t = new_a_t
        self.prev_b_t = new_b_t

        return d_t

    def diff_draw_down(self):
        del_p = self.close_price - self.prev_close_price
        r_t = del_p * self.action
        d_a = r_t - self.ddt_prev_a_t
        new_a_t = self.ddt_prev_a_t + 0.01 * d_a
        ddt_2 = self.prev_ddt ** 2
        new_dd_t = np.sqrt(ddt_2 + 0.01 * (min(r_t, 0) ** 2 - ddt_2))
        self.prev_ddt = self.prev_ddt if self.prev_ddt else 1
        if r_t > 0:
            d_t = (r_t - 0.5 * self.ddt_prev_a_t) / self.prev_ddt
        else:
            d_t = (ddt_2 * (r_t - 0.5 * self.ddt_prev_a_t) - 0.5 * self.ddt_prev_a_t * r_t ** 2) / (self.prev_ddt ** 3)

        self.prev_ddt = new_dd_t
        self.ddt_prev_a_t = new_a_t
        return d_t

    def multi_obj(self):
        daily_profit = np.mean(self.p_folio_diff)
        volatility = np.std(self.p_folio_diff)
        u = daily_profit + 0.01 * volatility
        return u

    def multi_obj_2(self):
        cumm = np.nancumsum(self.p_folio_diff)

        daily_profit = np.mean(cumm)
        volatility = np.std(cumm)
        u = self.alpha * daily_profit + self.beta * volatility
        return u

    def sharpe_ratio(self):
        # if self.index > 10:
        factor = 252 ** 0.5 if self.resolution == 'day' else 1
        return_ = pd.DataFrame(self.asset_memory).pct_change()
        _std = np.std(return_).item()
        if _std > 0:
            return float(factor * np.mean(return_) / _std)
        return 0

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({'date': date_list, 'account_value': asset_list})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        df_date = pd.DataFrame(self.date_memory)
        df_date.columns = ['date']

        action_list = self.action_memory
        close_list = self.close_memory
        df_actions = pd.DataFrame({'actions': action_list, 'close_price' : close_list})
        df_actions.index = df_date.date

        assert len(df_date) == len(df_actions), 'date and close price length must match actions length'
        return df_actions
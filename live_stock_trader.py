from envs import *
from models.model_register import *
import ray
from config import tech_inds 
from pickle5 import pickle
from ray.tune.registry import get_trainable_cls
from ray.tune.utils import merge_dicts
import logging
import pandas as pd
from downloader import add_tech_indicators
import os
from api_info import POLYGON_API
logger = logging.getLogger()


class LiveStockTrader():

    def __init__(self, resolution, symbol, lot, checkpoint_file, param_file):

        self.resolution = resolution
        self._api = POLYGON_API
        self.account = self._api.get_account()
        self._symbol = symbol
        self._lot = lot
        self._l = logger.getChild(self._symbol)

        self.prev_close = 0
        self.prev_action = 0
        self.obs = None

        now = pd.Timestamp.now(tz='America/New_York').floor('1min')
        market_open = now.replace(hour=9, minute=30)
        today = now.strftime('%Y-%m-%d')
        tomorrow = (now + pd.Timedelta('1day')).strftime('%Y-%m-%d')

        data = self._api.polygon.historic_agg_v2(symbol, 1, 'minute', today, tomorrow, unadjusted=False).df
        bars = data[market_open:]
        self._bars = pd.DataFrame(bars)

        self._state = 'NEW'
        self._agent, self._env = self.load_agent(checkpoint_file, param_file)
        self.state = self._agent.get_policy().model.get_initial_state()

    def get_balance(self):
        return float(self.account.cash)

    def load_agent(self, checkpoint_file, param_file):
        config_dir = os.path.dirname(checkpoint_file)
        config_path = os.path.join(param_file)
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
            # Load the config from pickled.
        else:
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        config['num_gpus'] = 0
        config['num_envs_per_worker'] = 0
        config['num_workers'] = 0
        if config['model']['custom_model'] == 'sdae_full':
            config['model']['custom_model_config']['train'] = False
        # config['explore'] = False
        ti = config['env_config']['tech_indicators']
        if ti in tech_inds.keys():
            config['env_config']['tech_indicators'] = tech_inds[ti]
        df = add_tech_indicators(self._bars, config['env_config']['tech_indicators'])
        config['env_config'] = merge_dicts(config['env_config'],
                                                (dict(resolution=self.resolution,
                                                      tickers=[self._symbol],
                                                      df_override=df,
                                                      max_shares=self._lot,
                                                      random_start=False,
                                                      initial_amount=self.account.cash,
                                                      test=True)))

        print('Trading lot:', self._lot)
        cls = get_trainable_cls('PPO') # todo: make automatic

        ray.init(include_dashboard=False)
        agent_ = cls(config=config)
        agent_.restore(checkpoint_file)
        _env = agent_.workers.local_worker().env
        return agent_, _env

    def _now(self):
        return pd.Timestamp.now(tz='America/New_York')

    def _outofmarket(self):
        return self._now().time() >= pd.Timestamp('15:55').time()

    def sell_stock(self, sell_qty):

        max_shares = self.get_balance() // self.prev_close
        # update balance
        # todo: we want to buy only what we can afford and reject other compared to just getting min
        calc_buy_qty = int(min(max_shares, sell_qty))

        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='market',
                qty=calc_buy_qty,
                time_in_force='day',
            )
        except Exception as e:
            self._l.info(e)
            self._transition('ATTEMPTED SELL')
            return

        self._l.info(f'submitted buy {order}')
        self._transition('BUY_SUBMITTED')

    def buy_stock(self, buy_qty):
        max_shares = self.get_balance() // self.prev_close

        calc_buy_qty = int(min(max_shares, buy_qty))

        try:
            order = self._api.submit_order(
                symbol=self._symbol,
                side='buy',
                type='market',
                qty=calc_buy_qty,
                time_in_force='day',
            )
        except Exception as e:
            self._l.info(e)
            self._transition('ATTEMPTED_BUY')
            return

        self._l.info(f'submitted buy {order}')
        self._transition('BUY_SUBMITTED')

    def step(self, bar):

        if self.state is None or self._bars is None:
            return

        self._bars = self._bars.append(pd.DataFrame({
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        }, index=[bar.start]))

        self._l.info(
            f'received bar start = {bar.start}, close = {bar.close}, len(bars) = {len(self._bars)}')

        if self._outofmarket():
            return

        self.update_state()

        res = self._agent.compute_action(observation=self._env.state, state=self.state, prev_action=self.prev_action, explore=False)

        if isinstance(res, tuple):
            action, self.state, _ = res
        else:
            action = res

        amount = int(self._env.scale_action(action))
        if amount > 0:
            self.buy_stock(amount)
        elif amount < 0:
            self.sell_stock(amount)
        else:
            self._transition('STOCKS HELD')

    def update_state(self):

        stocks_owned = 0
        for pos in self._api.list_positions():
            if pos.symbol == self._symbol:
                stocks_owned = int(pos.qty)

        self._env.stocks_owned = stocks_owned
        self._env.live_update(self._bars, self.get_balance(), stocks_owned)
        self._env.update_state()
        self.prev_close = self._env.close_price

    def _transition(self, new_state):
        self._l.info(f'transition from {self._state} to {new_state}')
        self._state = new_state



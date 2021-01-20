import argparse
import glob
import logging
import pickle

import tqdm

from api_info import POLYGON_API
from copy import copy
import yfinance as yf
import pandas as pd
import os
import multiprocessing as mp
import talib.abstract
import numpy as np
import talib
from talib.abstract import *
from talib import MA_Type
import itable
NY = 'America/New_York'
START = pd.Timestamp('2004-01-02 09:30', tz=NY).value / 1e6
END = pd.Timestamp('2020-12-31 15:00', tz=NY).value / 1e6

FINNHUB_API_KEY = 'bvcebsf48v6v1cifk3j0'

SAVE_DIR = os.path.join('datasets')


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """
    def __init__(self,
        start_date:str,
        end_date:str,
        ticker_list:list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df['tic'] = tic
            data_df=data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df=data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = ['date','open','high','low','close','adjcp','volume','tic']
            # use adjusted close price instead of close price
            data_df['close'] = data_df['adjcp']
            # drop the adjusted close price column
            data_df = data_df.drop('adjcp', 1)
        except NotImplementedError:
            print("the features are not supported currently")

        # convert date to standard string format, easy to filter
        data_df['date']=data_df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        #print("Display DataFrame: ", data_df.head())

        return data_df

    def select_equal_rows_stock(df):
        df_check=df.tic.value_counts()
        df_check=pd.DataFrame(df_check).reset_index()
        df_check.columns = ['tic','counts']
        mean_df = df_check.counts.mean()
        equal_list=list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df=df[df.tic.isin(select_stocks_list)]
        return df


def yahoo_data_downloader_multi(tickers):

    data = dict()
    for t in tickers:
        d = yf.download(t)
        if d is not None or not d.empty or len(d) > 0:
            data[t] = d
    return data


def polygon_data_downloader_multi(tickers, resolution, multiplier=1, stat_freq=20):
    if isinstance(tickers, str):
        args = [(ticker, resolution, multiplier, stat_freq) for ticker in tickers.split(' ')]
    else:
        args = [(ticker, resolution, multiplier, stat_freq) for ticker in tickers]

    with mp.Pool() as p:
        result = p.starmap(polygon_data_downloader, args)
        df_list = {tic[0]: df for tic, df in zip(args, result) if df is not None}

    return df_list


def polygon_data_downloader(ticker, resolution, multiplier=1, stat_freq=20):
    temp_start = START
    df_combined = pd.DataFrame()
    i = 0
    try:
        while temp_start < END:

            df = POLYGON_API.polygon.historic_agg_v2(ticker, multiplier, resolution, _from=temp_start, to=END).df
            if df.empty:
                logging.error(ticker, ':empty data')
                return None
            df_combined = df_combined.append(df)
            temp_start = df.index[-1].value / 1e6

            if df_combined.index[-1] == df_combined.index[-2]:
                break

            i += 1
            if i % stat_freq == 0:
                print(f'{ticker}: current start date: {df_combined.index[0]}, end date:'
                             f' {df_combined.index[-1]}, df_size: {len(df_combined)}')
    except Exception as e:
        logging.error(e)
        print('skipped', ticker)
        return None

    df_combined['tic'] = ticker

    return df_combined


def cache(df_dict, resolution, multiplier):
    dir_ = os.path.join(SAVE_DIR, f'{resolution}_{multiplier}')
    os.makedirs(dir_, exist_ok=True)
    for tic, df in df_dict.items():
        if len(df) < 1:
            continue
        path = os.path.join(dir_, f'{tic}_{df.index[0]}_{df.index[-1]}')
        path = path.replace(' ', '_')
        path = path.replace(':', '_')
        with open(path, 'wb') as pkl:
            pickle.dump(df, pkl)


def conv_list_int(x):
    if isinstance(x, list):
        return [int(v) for v in x]
    else:
        return int(x)


def add_tech_indicators(df, indicators, debug=False):

    if isinstance(indicators, str):
        indicators = indicators.split(' ')

    for ind in indicators:

        if ind == 'close_diff':
            close_diff = df['close'].diff()
            df['close_diff'] = np.tanh(close_diff)
        else:
            try:
                if debug:
                    print('adding', ind, 'to data')

                top = ind.split('!')
                assert len(top) <= 2, print('can only get one output or none')
                extra = '' if len(top) == 1 else top[1]
                splitted = top[0].split('(')
                assert len(splitted) != 0
                indicator = splitted[0]
                params = None if len(splitted) == 1 else splitted[1]
                ind_ = str(indicator).upper()
                fnc_a = getattr(talib.abstract, ind_)
                if params:
                    parameters = params[:-1].split('-')
                    flattened_params = [p.split(',') for p in parameters]
                    # convert to int
                    flattened_params = list(map(conv_list_int, flattened_params))
                    column_name = [ind_+'_'+p+extra for p in parameters]
                    if extra != '':
                        temp_dict = { c : fnc_a(df, *f)[extra] for c, f in zip(column_name, flattened_params) }
                    else:
                        temp_dict = {c: fnc_a(df, *f) for c, f in zip(column_name, flattened_params)}
                else:
                    if extra != '':
                        temp = fnc_a(df)[extra]
                    else:
                        temp = fnc_a(df)
                    temp_dict = {ind_:temp}

                df = pd.concat([df, pd.DataFrame(temp_dict)], axis=1)

            except KeyError as e:
                print(e)

    df = df.bfill().ffill()

    if np.any(pd.isna(df)):
        df = df.fillna(0)

    return df


def load(tic, resolution='minute_1'):
    dir_ = os.path.join(SAVE_DIR, f'{resolution}')
    files = glob.glob(os.path.join(dir_, f'{tic}_*'))

    assert len(files) > 0, f'{tic}: ticker data has not been downloaded'
    file = files[0]

    with open(file, 'rb') as pkl:
        df = pickle.load(pkl)
    return df


def load_multiple(tic, resolution='minute_1'):
    if isinstance(tic, str):
        tic = tic.split(' ')
    df_dict = {}
    for t in tic:
        df_dict[t] = load(t, resolution)

    return df_dict


def load_dataset(tickers, tech_indicators=None, resolution='minute_1'):
    df_dict = load_multiple(tickers, resolution)

    if tech_indicators:
        for k, v in df_dict.items():
            if tech_indicators == 'ALL':
                for group, funcs in talib.get_function_groups().items():
                    if group == 'Volume Indicators' or\
                        group == 'Volatility Indicators' or\
                        group == 'Overlap Studies' or \
                        group == 'Momentum Indicators':
                        print(group)
                        print('-----------------------------------------')

                        for func in tqdm.tqdm(funcs):
                            if func == 'MAVP':
                                continue
                            f = Function(func)
                            t_df = f(v)
                            if len(f.info['output_names']) > 1:
                                for o in f.info['output_names']:
                                    v[o] = t_df[o]
                            else:
                                v[func] = t_df
                            df_dict[k] = v
                df_dict[k] = df_dict[k].bfill().ffill()
                if np.any(pd.isna(df_dict[k])):
                    df_dict[k] = df_dict[k].fillna(0)
            else:
                df_dict[k] = add_tech_indicators(v, tech_indicators)

    return df_dict


from config import S_P
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', '-r', default='minute')
    parser.add_argument('--ticker', '-t', required=True)

    arg = parser.parse_args()

    if arg.resolution == 'day':
        if arg.ticker == 'S_P':
            data = yahoo_data_downloader_multi(tickers=S_P)
        else:
            data = yahoo_data_downloader_multi(tickers=arg.ticker)
        cache(data, 'day', 1)
    else:
        if arg.ticker == 'S_P':
            batch = min(len(S_P), os.cpu_count())
            for i in range(0, len(S_P), batch):
                res = polygon_data_downloader_multi(S_P[i:i + batch], arg.resolution, stat_freq=20)
                cache(res, arg.resolution, 1)
        else:
            res = polygon_data_downloader_multi(arg.ticker, arg.resolution, stat_freq=20)
            cache(res, arg.resolution, 1)

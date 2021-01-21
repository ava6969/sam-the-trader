import logging
import math
import os
import glob
import signal
import sys
import threading
import ray
from colorama import Fore
from pickle5 import pickle
from ray.tune.registry import get_trainable_cls
import tqdm
from tabulate import tabulate
import time
from models.model_register import *
from envs import *
from backtest import BackTestStats
import argparse
import colorama
from ray.tune.utils import merge_dicts
from torch.multiprocessing import Pool, set_start_method


def handler(signal, frame):
    logging.info(f'{frame} : {signal} caught ')
    ray.shutdown()
    sys.exit()


RESULT_DIR = f'results'


def create_agent(run, tick_to_trade, tf, passive, checkpoint, param_file, initial_amount_over_, h_max_over_,
                 start_date_, end_date_, _log_every):
    """
    Restores and Creates a  trading agent
    Parameters
    ----------
    run: algorithm to use
    tick_to_trade: the ticker being traded
    tf: timeframe
    passive: if log is enabled or not
    checkpoint: checkpoint file
    param_file
    initial_amount_over_: override capital used in training
    h_max_over_: override maximum shares allowed for trading by agent
    start_date_: date to start trading from
    end_date_: date to end trade
    _log_every: if not passive log every n amount

    Returns
    -------

    """
    config_path = os.path.join(param_file)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    config['num_gpus'] = 1
    config['num_envs_per_worker'] = 0
    config['num_workers'] = 0
    if config['model']['custom_model']=='sdae_full':
      config['model']['custom_model_config']['train'] = False
    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = False
    initial_amount = config["env_config"]["initial_amount"] if not initial_amount_over_ else initial_amount_over_
    h_max = config["env_config"]["max_shares"]  if not h_max_over_ else h_max_over_

    log_every = math.inf if passive else _log_every
    config['env_config'] = merge_dicts(config['env_config'],
                                       (dict(log_every=log_every, resolution=tf,
                                             tickers=[tick_to_trade],
                                             max_shares=h_max, random_start=False,
                                             filter_date=False,
                                             initial_amount=initial_amount, test=True, start_date=start_date_,
                                             end_date=end_date_)))

    print('Trading lot:', h_max)
    cls = get_trainable_cls(run)

    agent_ = cls(config=config)
    agent_.restore(checkpoint)
    return agent_, config, initial_amount, h_max


def interact(tf, tick, tick_to_trade, s_date, e_date, init_amt=None, init_h=None, passive=False, log_every=math.inf):

    keys = gather_keys()
    runs = []
    for i, key in enumerate(keys):
        res = key.split('_')
        config = '_'.join(res[1:-1])
        runs.append(res[1].split('-')[0])
        print('[{}] tic = {} config - {} date - {}'.format(i, res[0], config, res[-1]))

    choice = int(input(f'Select from 0 - {len(keys) - 1} >> '))
    key = keys[choice]

    choice2 = input('evaluate all ? y/n')
    if choice2 == 'y':
        evaluate_key(tf, runs[choice], key, tick, tick_to_trade, s_date, e_date, init_amt, init_h, passive, log_every, False)
        return

    param_files, progress_df, checkpoints, date = gather_checkpoint_files(key)
    print('Selected ', key)

    extra, param_file, checkpoint = select_checkpoint(checkpoints, progress_df, param_files)
    ray.init()
    agent, config, initial_amount, h_max = create_agent(runs[choice], tick_to_trade, tf, passive, checkpoint, param_file,
                                                        init_amt, init_h, s_date, e_date, _log_every=log_every)

    trade(key, agent, config, date, tick, tick_to_trade, initial_amount, h_max, extra)
    ray.shutdown()


def evaluate_run(tf, run, key, checkpoint, param_file, extra, passive, init_amt, init_h, tick, tick_to_trade, date, s_date, e_date, log_every):
    ray.init(ignore_reinit_error=True)
    try:
        agent, config, initial_amount, h_max = create_agent(run, tick_to_trade, tf, passive, checkpoint,
                                                            param_file,
                                                            init_amt, init_h, s_date, e_date,
                                                            _log_every=log_every)
        trade(key, agent, config, date, tick, tick_to_trade, initial_amount, h_max, extra)
    except Exception as e:
        logging.error(e)
        sys.exit()
    finally:
        ray.shutdown()


def evaluate_key(tf, run, key, tick, tick_to_trade, s_date, e_date, init_amt=None, init_h=None, passive=True,  log_every=math.inf, multithread=True):
    param_files, progress_df, checkpoints, date = gather_checkpoint_files(key)
    args = []
    for i in range(len(checkpoints)):
        extra = checkpoints[i].split('/')
        extra = extra[-1]
        args.append((tf, run, key, checkpoints[i], param_files[0], extra,
                     passive, init_amt, init_h, tick, tick_to_trade, date, s_date, e_date, log_every))
    batch = 9
    if multithread:
        for i in range(0, len(checkpoints), batch):
            threads = [threading.Thread(target=evaluate_run, args=args[j + i]) for j in range(batch)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    else:
        for i in range(0, len(checkpoints), batch):
            with Pool() as pool:
                pool.starmap(evaluate_run, args[i:i+batch])


def select_checkpoint(checkpoints, progress_df, param_files, checkpoint_idx=None, eval_all=False):
    print('//////// CHECKPOINTS /////////')
    checkpoint_df = pd.DataFrame()

    for i, ckp in enumerate(checkpoints):
        iter = int(ckp.split('-')[-1]) - 1
        rew,  pl, vl, e = None, None, None, None
        if iter < len(progress_df):
            rew = progress_df.iloc[iter].episode_reward_mean
        pl = progress_df.iloc[iter]['info/learner/default_policy/policy_loss']
        vl = progress_df.iloc[iter]['info/learner/default_policy/vf_loss']
        _e = progress_df.iloc[iter]['info/learner/default_policy/entropy']
        c = ckp.split('/')[-1]
        n = int(c.split('-')[-1])
        checkpoint_df = checkpoint_df.append(pd.Series([n, rew, pl, vl, _e]), ignore_index=True)

    checkpoint_df.columns = ['checkpoint_id', 'mean_reward', 'policy_loss',
                             'value_loss', 'entropy']

    c_idx_max = checkpoint_df.checkpoint_id.idxmax()
    c = checkpoint_df.checkpoint_id[c_idx_max]
    checkpoint_df.checkpoint_id[c_idx_max] = str(c)

    c_idx_max = checkpoint_df.mean_reward.idxmax()
    r = checkpoint_df.mean_reward[c_idx_max]
    checkpoint_df.mean_reward[c_idx_max] = Fore.GREEN + str(r) + Fore.RESET

    c_idx_max = checkpoint_df.policy_loss.idxmin()
    p = checkpoint_df.policy_loss[c_idx_max]
    checkpoint_df.policy_loss[c_idx_max] = Fore.YELLOW + str(p) + Fore.RESET

    c_idx_max = checkpoint_df.value_loss.idxmin()
    v = checkpoint_df.value_loss[c_idx_max]
    checkpoint_df.value_loss[c_idx_max] = Fore.LIGHTMAGENTA_EX + str(v) + Fore.RESET

    c_idx_max = checkpoint_df.entropy.idxmin()
    e = checkpoint_df.entropy[c_idx_max]
    checkpoint_df.entropy[c_idx_max] = Fore.LIGHTRED_EX + str(e) + Fore.RESET

    checkpoint_df.reset_index(inplace=True)

    if checkpoint_idx is not None or eval_all:
        checkpoint = checkpoints[checkpoint_idx]
        row = checkpoint_df.iloc[checkpoint_idx]
    else:

        table = tabulate(
            headers=checkpoint_df.columns,
            tabular_data=checkpoint_df.values.tolist(), tablefmt="fancy_grid")
        print(table)
        choice2 = int(input(f'Select index >> '))
        checkpoint = checkpoints[choice2]

        row = checkpoint_df.iloc[choice2]

    extra = 'checkpoint-' + str(int(row[1]))
    if isinstance(row[2], str):
        extra += '_MAX_REWARD'
    if isinstance(row[3], str):
        extra += '_MIN_POLICY_LOSS'
    if isinstance(row[4], str):
        extra += '_MIN_VALUE_LOSS'
    if isinstance(row[5], str):
        extra += '_MIN_ENTROPY'

    logging.info('Checkpoint Name: ', extra)
    assert len(param_files) > 0
    param_file = ''
    for p in param_files:
        if p.split('/')[2] == checkpoint.split('/')[2]:
            param_file = p
            break

    logging.info('Selected pickle file: ', param_file)
    return extra, param_file, checkpoint


def trade(key, agent_, config, date, tick, tick_to_trade, initial_amount, h_max, extra):
    env_id = config['env']
    trade_env = agent_.workers.local_worker().env
    print(env_id)
    # sys.stdout.flush()
    start = time.time()
    account_memory = []
    actions_memory = []
    action = trade_env.action_space.sample()
    test_obs = trade_env.reset()
    length = len(trade_env.train_data)
    state = agent_.get_policy().model.get_initial_state()
    total_reward = 0

    for i in range(length):
        res = agent_.compute_action(observation=test_obs, state=state, prev_action=action)
        if isinstance(res, tuple):
            action, state, _ = res
        test_obs, reward, _, info = trade_env.step(action)
        total_reward += reward
        if i == length - 2:
            account_memory = trade_env.save_asset_memory()
            actions_memory = trade_env.save_action_memory()
    end = time.time()
    print(f'Trading time: ', (end - start) / 60, ' minutes Reward:', total_reward)
    sys.stdout.flush()

    df_account_value = account_memory
    extra_ = f'{env_id}_{date}_{tick}_{tick_to_trade}_{initial_amount}_{h_max}_{extra}_{total_reward}'

    os.makedirs(f'./backtest_accounts/{key}',exist_ok=True)

    df_account_value.to_csv(f"./backtest_accounts/{key}/df_acc_val_{extra_}.csv")
    actions_memory.to_csv(f"./backtest_accounts/{key}/df_act_val_{extra_}.csv")


def gather_keys():
    """
    Gather all keys or model trainign data in result folder
    Returns
    -------

    """
    keys_dir = glob.glob(RESULT_DIR + f'/*')

    if not len(keys_dir):
        logging.info('No checkpoint-20 saved')
        return
    keys = [res.split('/')[-1] for res in keys_dir]

    return keys


def gather_checkpoint_files(key):
    """
    Gather all checkpoints associated to a key or checkpoint folder
    Parameters
    ----------
    key: checkpoint folder

    Returns
    -------

    """
    param_files = glob.glob(f'results/{key}/*/*.pkl')
    progress_files = glob.glob(f'results/{key}/*/progress.csv')
    progress_df_list = [pd.read_csv(p) for p in progress_files]
    progress_df = pd.concat(progress_df_list)

    path = os.path.join(RESULT_DIR, key, '*', 'checkpoint_*', 'checkpoint-*')
    checkpoints = glob.glob(path)
    checkpoints = [v for v in checkpoints if v[-8:] != 'metadata']
    date = key.split('_')[1]

    return param_files, progress_df, checkpoints, date


def resample_min(df_account_value):
    """
    Resamples minute date to day
    Parameters
    ----------
    df_account_value: account information to resample

    Returns
    -------

    """
    df_account_value.set_index('date', drop=True, inplace=True)
    datetime_series = pd.to_datetime(df_account_value.index,utc=True)
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    df_account_value.set_index(datetime_index, drop=True, inplace=True)
    res = df_account_value.resample('B', convention='end').mean()
    df_account_value = res.reset_index()
    return df_account_value


def update_data():
    dir = glob.glob('backtest_accounts/*/df_acc_val*.csv')
    dir_ = glob.glob('backtest_accounts/*/df_act_val*.csv')
    res = pd.DataFrame()
    account_values = []
    actions_close = []
    for i, file in tqdm.tqdm(enumerate(dir)):
        v = file.split('_')
        df_account_value = pd.read_csv(dir[i]).reset_index(drop=True)
        df_account_value = resample_min(df_account_value)
        df_account_value = df_account_value.drop('Unnamed: 0', axis=1)
        df_account_value.columns = ['date', 'account_value']
        account_values.append(df_account_value)
        actions_close.append(pd.read_csv(dir_[i]).reset_index(drop=True))
        experiment = BackTestStats(account_value=df_account_value, debug=False)
        t, ttt, amt, h_m, id = v[8:13]
        id = id.split('-')[-1]
        rew = v[-1]
        temp = pd.Series(data=[int(id), ttt, float(amt), float(h_m), float(rew[:-4])],
                         index=['checkpoint', 'TickTraded', 'Initial Amount', 'H_MAX',
                                'mean_reward'])
        experiment = experiment.append(temp)
        res = res.append(experiment, ignore_index=True)

    res.drop(['Skew', 'Kurtosis'], axis=1, inplace=True)
    # res = res.reset_index()
    return dir, account_values, actions_close, res


if __name__ == '__main__':
    colorama.init(autoreset=True)
    try:
        set_start_method('forkserver')
    except RuntimeError as e:
        print(e)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', '-t', default='minute_1')
    parser.add_argument('--start_date', '-b', default='2019-12-01 09:30:00')
    parser.add_argument('--end_date', '-e', default='2020-12-01 17:31:00')
    parser.add_argument('--log_every', '-r', default=5000, required=False)
    parser.add_argument('--tick', '-s', default='AAPL')
    parser.add_argument('--tick_trade', '-u', default='AAPL')
    parser.add_argument('--init_amt', '-i', default=25000)
    parser.add_argument('--h_max', '-l', default='3')

    mut = parser.add_mutually_exclusive_group()
    mut.add_argument('--passive', '-p', action='store_true')

    signal.signal(signal.SIGINT, handler)
    from signal import signal, SIGPIPE, SIG_DFL
    signal(SIGPIPE, SIG_DFL)

    args = parser.parse_args()
    tf = args.tf
    start_date = args.start_date
    end_date = args.end_date
    tick = args.tick
    h_max_over = None if not args.h_max else float(args.h_max)
    initial_amount_over = None if not args.init_amt else float(args.init_amt)
    tick_trade = tick if not args.tick_trade else args.tick_trade
    log_every = args.log_every

    interact(tf, tick, tick_to_trade=tick_trade,
             s_date=start_date, e_date=end_date,
             init_amt=initial_amount_over,
             init_h=h_max_over, log_every=log_every,
             passive=args.passive)

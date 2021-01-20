import argparse
import datetime
from models.model_register import *
import os
from api_info import RESTORE_PATH
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
import ray
import yaml
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.utils import merge_dicts
from envs import *


def train(name, ray_config, debug=False):
    """
    Trains sam
    Parameters
    ----------
    name: name of yaml file
    ray_config: ray configuration
    debug: whether to test in editor

    Returns
    -------

    """
    ray.init()
    trainer_class = get_trainable_cls(ray_config['run'])
    default_config = trainer_class._default_config.copy()
    config = merge_dicts(default_config, ray_config['config'])
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    t = ray_config['config']['env_config']['tickers']
    run = ray_config['run']

    if config['model']['custom_model'] and config['model']['custom_model'][:4] == 'sdae':
        config['callbacks'] = MyCallbacks
    _t = '|'.join(t)
    model_name = f'{_t}_{name}_{now}'
    print(f'\33]0;{model_name} - {name}\a', end='', flush=True)
    if debug:
        config['num_workers'] = 0
        config['num_envs_per_worker'] = 1
        # config['train_batch_size'] = 10
        config['env_config']['log_every'] = 2000
        trainer = trainer_class(config=config)
        policy = trainer.get_policy()
        model = policy.model
        print(model)
        for i in range(10):
            res = trainer.train()
            print(pretty_print(res))
    else:
        tune.run(
            run,
            name=model_name,
            stop=ray_config['stop'],
            local_dir='results',
            config=config,
            checkpoint_at_end=True,
            verbose=0,
            restore=RESTORE_PATH,
            checkpoint_freq=ray_config['checkpoint_freq'])

    ray.shutdown()


class MyCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        model = trainer.get_policy().model
        loss = model.sdae_loss
        result["custom_metrics"]["sdae_loss"] = loss


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", '-c', required=True)
    arg_parser.add_argument("--debug", "-d", action='store_true')
    arg_parser.add_argument("--restore_path", '-r', default='')

    args = arg_parser.parse_args()

    yaml_file = os.path.join('configs', f'{args.config}.yaml')
    parsed = yaml.load(open(yaml_file, 'rb'), yaml.Loader)
    ray_config = parsed['runner']
    train(args.config, ray_config, args.debug)


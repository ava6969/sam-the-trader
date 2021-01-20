from sim_stock_trader import SimStockTrader
from wrappers import *
from gym import register
from ray.tune import register_env


def env_creator(config):

    arg_config = copy(config)
    wrappers = []
    if 'feature_extractors' in config.keys():
        del arg_config['feature_extractors']
        wrapper_arg = config['feature_extractors']
        wrappers = wrapper_arg.split()

    env = SimStockTrader(**arg_config,
                         worker_index=config.worker_index,
                         vector_index=config.vector_index)
    for wrapper in wrappers:
        args = wrapper.split('_')
        if args[0].lower() == 'stack':
            print('wrapping with stack: ', args[1])
            if len(args) > 0:
                arg_ = int(args[1])
                env = StackWrapper(env, arg_)
            else:
                env = StackWrapper(env)
        elif args[0].lower() == 'kline':
            print('wrapping with kline')
            env = KLineWrapper(env)
        elif args[0].lower() == 'gauss':
            print('wrapping with gauss')
            env = GaussianCorruptObservation(env)
        elif args[0].lower() == 'norm':
            print('wrapping with norm')
            env = NormWrapper(env)
        elif args[0].lower() == 'numpy':
            print('wrapping with numpy')
            env = NumpyWrapper(env)

    if len(wrappers) == 0:
        print('wrapping with numpy')
        env = NumpyWrapper(env)

    return env


register_env('BasicStockTrader-v0', lambda config: env_creator(config))
register(id='BasicStockTrader-v0', entry_point='envs:SimStockTrader')

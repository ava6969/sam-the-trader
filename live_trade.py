import argparse
import asyncio
import signal
import sys
import logging
import alpaca_trade_api as alpaca
from api_info import API_KEY, API_SECRET, CHECKPOINT_FILE, PARAM_FILE
logger = logging.getLogger()
from live_stock_trader import *


def handler(signal, frame):
    print(f'{frame} : {signal} caught ')
    ray.shutdown()
    sys.exit()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--tf", '-t', default=1)
    arg_parser.add_argument("--symbol", '-s', default='AAPL')
    arg_parser.add_argument("--lot", '-l', default=3)

    fmt = '%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler('console.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    args = arg_parser.parse_args()
    symbol = args.symbol
    qc = 'Q.%s' % symbol
    tc = 'T.%s' % symbol
    # trader = LiveTrader(symbol, args.lot, args.algo, args.tf)
    conn = alpaca.StreamConn(key_id=API_KEY, secret_key=API_SECRET, data_stream='polygon', debug=True)
    for sig in [signal.SIGINT]:
        signal.signal(sig, handler)

    trader = LiveStockTrader(args.tf, args.symbol, args.lot, CHECKPOINT_FILE, PARAM_FILE)

    @conn.on(r'Q$')
    async def on_quote(conn, channel, data):
        # Quote update received
        print(data)

    @conn.on(r'^AM')
    async def on_bars(conn, channel, data):
        trader.step(data)

    @conn.on(r'trade_updates')
    async def on_trade_updates(conn, channel, data):
        logger.info(f'trade_updates {data}')

    @conn.on(r'^status$')
    async def on_status(conn, channel, data):
        logger.info(f'polygon status update {data}')


    async def periodic():
        while True:
            if not trader._api.get_clock().is_open:
                logger.info('exit as market is not open')
                sys.exit(0)
            await asyncio.sleep(30)

    channels = ['AM.' + symbol, 'trade_updates']
    loop = conn.loop

    try:
        loop.run_until_complete(asyncio.gather(
            conn.subscribe(channels),
            periodic(),
        ))
        loop.close()
    finally:
        ray.shutdown()

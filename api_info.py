import os
import alpaca_trade_api as tradeapi

API_KEY = "PKSL8S3NYPVTUJSNYALZ"
API_SECRET = "ICm51ekdzAvVk7rEToYSrAzptAZReMVkvCuksH0m"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

os.environ.setdefault('APCA_API_KEY_ID', API_KEY)
os.environ.setdefault('APCA_API_SECRET_KEY', API_SECRET)
os.environ.setdefault('APCA_API_BASE_URL', APCA_API_BASE_URL)
os.environ.setdefault('APCA_RETRY_MAX', '10')
os.environ.setdefault('APCA_RETRY_WAIT', '10')


POLYGON_API = tradeapi.REST()
RESTORE_PATH = 'results/R|L|O|L|-|8_PPO_20210119-15h19/PPO_BasicStockTrader-v0_9cb01_00000_0_2021-01-19_15-19-20/checkpoint_180/checkpoint-180'
CHECKPOINT_FILE = os.path.join('active_model', 'checkpoint-180')
PARAM_FILE = os.path.join('active_model', 'params.pkl')

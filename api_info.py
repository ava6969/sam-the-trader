import os
import alpaca_trade_api as tradeapi

API_KEY = "PK1Y132GWLQG53SDU6NM "
API_SECRET = "oxQk2Lqz486HL8g0iSGcOgGqwcVsn5twcmhfUCqh"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

os.environ.setdefault('APCA_API_KEY_ID', API_KEY)
os.environ.setdefault('APCA_API_SECRET_KEY', API_SECRET)
os.environ.setdefault('APCA_API_BASE_URL', APCA_API_BASE_URL)
os.environ.setdefault('APCA_RETRY_MAX', '10')
os.environ.setdefault('APCA_RETRY_WAIT', '10')


POLYGON_API = tradeapi.REST()
RESTORE_PATH = ''
CHECKPOINT_FILE = os.path.join('active_model', 'checkpoint-180')
PARAM_FILE = os.path.join('active_model', 'params.pkl')

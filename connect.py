import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()
client = TradingClient(os.getenv("APCA_API_KEY_ID"),
                       os.getenv("APCA_API_SECRET_KEY"), paper=True)

acct = client.get_account()
print("Status:", acct.status)
print("Cash:", acct.cash)
print("Buying power:", acct.buying_power)
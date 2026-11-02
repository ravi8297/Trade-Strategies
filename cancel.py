import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv()
client = TradingClient(os.getenv("APCA_API_KEY_ID"),
                       os.getenv("APCA_API_SECRET_KEY"), paper=True)

open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
for o in open_orders:
    print("Open:", o.symbol, o.side, o.qty, o.status)

client.cancel_orders()
print("Cancelled all open orders.")
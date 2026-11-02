import os, sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

load_dotenv()
client = TradingClient(os.getenv("APCA_API_KEY_ID"),
                       os.getenv("APCA_API_SECRET_KEY"), paper=True)

side = OrderSide.BUY if sys.argv[1].lower() == "buy" else OrderSide.SELL
symbol, qty = sys.argv[2].upper(), int(sys.argv[3])

order = client.submit_order(MarketOrderRequest(
    symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY))
print("Order sent:", order.id, order.status)

for p in client.get_all_positions():
    print(p.symbol, "qty", p.qty, "P/L", p.unrealized_pl)
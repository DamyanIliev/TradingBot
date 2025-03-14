from binance.um_futures import UMFutures
from datetime import datetime
import ta
from ta.trend import SMAIndicator
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time
from binance.error import ClientError
from playsound3 import playsound

# Binance API keys
api = 'eVJmzdRHtvD7cduRQ4YEfbLGT3MelccyA7eGkbRgWS2U0DMnxVknDVFkNVKbrvGp'
secret = 'OdHnel7a2DhJFFafREQlJhoHmJWSOAejKlqoyJHvHi7QaF9yt2H7BOWHsFsjBFBm'

client = UMFutures(key=api, secret=secret)

volume = 125  # volume for one order (if its 10 and leverage is 10, then you put 1 usdt to one position)
leverage = 25
qty = 1  # Amount of concurrent opened positions

def play_sound():
    playsound("trading_bot_audio.mp3")

# Get account balance
def get_balance_usdt():
    try:
        response = client.balance(recvWindow=6000)
        for elem in response:
            if elem['asset'] == 'USDC':
                return float(elem['balance'])
    except ClientError as error:
        print(f"API Error: {error.error_message}")


# Get symbol tickers
def get_tickers_usdt():
    return [t['symbol'] for t in client.ticker_price() if 'USDT' in t['symbol']]

def get_tickers_specific():
    allowed_symbols = {"TRXUSDT", "ADAUSDT", "VETUSDT", "BTCUSDT", "ETHUSDT",
                       "PNUTUSDT", "FARTCOINUSDT", "XLMUSDT", "XRPUSDT", "LTCUSDT"}

    return [t['symbol'] for t in client.ticker_price() if t['symbol'] in allowed_symbols]


# Fetch candle data
def klines(symbol, interval):
    try:
        df = pd.DataFrame(client.klines(symbol, interval))
        df = df.iloc[:, :6]
        df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df.set_index('Time', inplace=True)
        return df.astype(float)
    except ClientError as error:
        print(f"API Error: {error.error_message}")

def rma(series, length):
    """Calculates the Rolling Moving Average (RMA) manually."""
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

# New Indicators Implementation
def get_technical_indicators(symbol, interval):
    df = klines(symbol, interval)

    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['EMA200_High'] = ta.trend.ema_indicator(df['High'], window=200)  # EMA based on High prices
    df['EMA200_Low'] = ta.trend.ema_indicator(df['Low'], window=200)  # EMA based on Low prices

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'],
    df['Volume']).volume_weighted_average_price()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    # Set Bollinger Bands length to 30
    bb_30 = ta.volatility.BollingerBands(df['Close'], window=30)
    df['BB_Upper_30'] = bb_30.bollinger_hband()
    df['BB_Middle_30'] = bb_30.bollinger_mavg()
    df['BB_Lower_30'] = bb_30.bollinger_lband()
    df['BB_Width_30'] = (df['BB_Upper_30'] - df['BB_Lower_30']) / df['BB_Middle_30']

    length = 110
    df['EMA_High_110'] = ta.trend.ema_indicator(df['High'], window=length)
    df['EMA_Low_110'] = ta.trend.ema_indicator(df['Low'], window=length)
    df['RMA_High_110'] = rma(df['High'], length=length)
    df['RMA_Low_110'] = rma(df['Low'], length=length)

    # ➤ Detect trends for Single & Double Cloud
    df['is_bullish_single'] = (df["EMA_High_110"] > df["EMA_High_110"].shift(1)) & (
                df["EMA_Low_110"] > df["EMA_Low_110"].shift(1))
    df['is_bearish_single'] = (df["EMA_High_110"] < df["EMA_High_110"].shift(1)) & (
                df["EMA_Low_110"] < df["EMA_Low_110"].shift(1))

    # df['bullish_double'] = (df["EMA_High_110"] > df["RMA_High_110"]) & (df["EMA_Low_110"] > df["RMA_Low_110"])
    # df['bearish_double'] = (df["EMA_High_110"] < df["RMA_High_110"]) & (df["EMA_Low_110"] < df["RMA_Low_110"])

    return df

# Display all columns and rows
# df = get_technical_indicators("TRXUSDT", "15m")
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.max_rows', 10)  # Adjust rows as needed (set None to see all)
# print(df.tail(20))
def get_price_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['pricePrecision']

# Amount precision. BTC has 3, XRP has 1
def get_qty_precision(symbol):
    resp = client.exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['quantityPrecision']

def set_leverage(symbol, leverage):
    try:

        # Set leverage for the symbol
        response_leverage = client.change_leverage(symbol=symbol, leverage=leverage)
        print(f"Leverage set to {leverage} for {symbol}. Response: {response_leverage}")
    except ClientError as error:
        print(f"Error setting leverage: {error.error_message}")

# Refined order placing with error handling
def open_order_with_error_handling(symbol, side ,interval):
    side = side.upper()  # Convert to uppercase for consistency
    set_leverage(symbol, leverage)

    if side not in ['BUY', 'SELL']:
        print(f"Invalid side: {side}. It should be 'BUY' or 'SELL'.")
        return

    price = float(client.ticker_price(symbol)['price'])  # Get current market price
    qty_precision = get_qty_precision(symbol)
    price_precision = get_price_precision(symbol)
    qty = round(volume / price, qty_precision)

    # Get latest ATR value for SL & TP calculation
    df = get_technical_indicators(symbol, interval)
    atr = df['ATR'].iloc[-1]

    sl_mult = 1.5  # Stop-Loss Coefficient
    tp_mult = 2  # Take-Profit Coefficient

    if side == 'BUY':
        stop_loss = round(price - (sl_mult * atr), price_precision)
        take_profit = round(price + (tp_mult * atr), price_precision)
    else:  # side == 'SELL'
        stop_loss = round(price + (sl_mult * atr), price_precision)
        take_profit = round(price - (tp_mult * atr), price_precision)

    print(f"{interval} : {side} signal for {symbol} at price {price}")
    print(f"{interval} : Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")

    try:
        # Place main order
        order_response = client.new_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            quantity=qty,
            timeInForce='GTC',
            price=price
        )
        print(f"Main Order placed: {order_response}")

        # Place separate Stop-Loss order
        sl_order_response = client.new_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type="STOP_MARKET",
            quantity=qty,
            stopPrice=stop_loss,
            timeInForce='GTC',
            closePosition=True
        )
        print(f"Stop-Loss Order placed: {sl_order_response}")

        # Place separate Take-Profit order
        tp_order_response = client.new_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type="TAKE_PROFIT_MARKET",
            quantity=qty,
            stopPrice=take_profit,
            timeInForce='GTC',
            closePosition=True
        )
        print(f"Take-Profit Order placed: {tp_order_response}")
        play_sound()

    except Exception as e:
        print(f"Error placing order: {e}")

def generate_signal_bb(symbol, interval):
    df = get_technical_indicators(symbol, interval)
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    #latest['MA10_High'] > latest['EMA20_High'] and and latest['Close'] > latest['MA6'] and pd.notna(latest['SAR_Up']) and pd.notna(previous['SAR_Up'])
    if previous['Close'] < previous['BB_Lower_30'] and previous['RSI'] < 30 and latest['Close'] > previous['High'] and latest['BB_Width_30'] > 0.003 and latest['Close'] > latest['BB_Lower_30'] and latest['is_bullish_single']:
        return 'BUY'

    #latest['MA10_Low'] < latest['EMA20_Low'] and and latest['Close'] < latest['MA6'] and pd.notna(latest['SAR_Down']) and pd.notna(previous['SAR_Down'])
    if previous['Close'] > previous['BB_Upper_30'] and previous['RSI'] > 70 and latest['Close'] < previous['Low'] and latest['BB_Width_30'] > 0.003 and latest['Close'] < latest['BB_Upper_30'] and  latest['is_bearish_single']:
        return 'SELL'

    return 'none'

def check_open_position(symbol):
    try:
        positions = client.get_position_risk()  # Get all open positions
        for position in positions:
            if position['symbol'] == symbol and abs(float(position['positionAmt'])) > 0:
                return True  # There is an open position (long or short)
        return False  # No open position found
    except Exception as e:
        print(f"Error checking open position for {symbol}: {e}")
        return False

symbols = get_tickers_usdt()

def process_symbol_interval(symbol, interval):
    """Function to check signals and open orders for a symbol and interval."""
    signal = generate_signal_bb(symbol, interval)
    if signal == 'BUY':
        print(f"{interval}: BUY signal detected for {symbol} at {datetime.now()}")
        open_order_with_error_handling(symbol, 'buy', interval)
    elif signal == 'SELL':
        print(f"{interval}: SELL signal detected for {symbol} at {datetime.now()}")
        open_order_with_error_handling(symbol, 'sell', interval)

while True:
    try:
        balance = get_balance_usdt()
        btc_position_open = check_open_position("BTCUSDT")
        print("────────────────────────────────")
        print(f"Balance: {balance:.2f} USDC")
        print("BTCUSDT position is open!" if btc_position_open else "No open position for BTCUSDT.")

        if balance is None:
            print("⚠️ API Connection Issue. Retrying...")
        else:
            # Run symbol processing in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for symbol in symbols:
                    for interval in ['1m']:
                        futures.append(executor.submit(process_symbol_interval, symbol, interval))

                # Wait for all tasks to complete
                for future in futures:
                    future.result()

        print("Waiting for next cycle...\n")
        time.sleep(50)  # Sleep 4 minutes before the next run

    except Exception as e:
        print(f"⚠️ Error in main loop: {e}")
        time.sleep(10)
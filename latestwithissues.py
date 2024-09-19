import pandas as pd
import ta
from binance.client import Client
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from dateutil.relativedelta import relativedelta
from datetime import datetime
import time
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import asyncio
import httpx
import telegram
from datetime import timedelta
from telegram.ext import Updater, CommandHandler, MessageHandler
import numpy as np
from datetime import timedelta
import feedparser
import joblib
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

analyzer = SentimentIntensityAnalyzer()  # Initialize globally

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def get_recent_headlines(symbol):
    """
    Fetches recent news headlines for a given cryptocurrency symbol from CoinDesk RSS feed.
    """
    try:
        rss_url = 'https://www.coindesk.com/feed/'
        feed = feedparser.parse(rss_url)
        headlines = [entry.title for entry in feed.entries if symbol.lower() in entry.title.lower()]
        return headlines
    except Exception as e:
        print(f"Error fetching news headlines: {e}")
        return []

def fetch_news_sentiment(symbol, analyzer):
    """
    Fetches sentiment scores for the news headlines related to the given symbol.
    Weighs recent news more heavily.
    """
    headlines = get_recent_headlines(symbol)
    if not headlines:
        return 0
    # Weigh more recent news more heavily
    time_decay_factor = np.linspace(1.0, 0.1, len(headlines))
    # Calculate sentiment score for each headline
    scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    # Compute weighted average
    weighted_sentiment = np.average(scores, weights=time_decay_factor)
    
    return weighted_sentiment
# Initialize Binance client
api_key = 'cLVgIkkwrnKDoOJtBzEUVo2fP7a09fVhFLDVl1A4j6LosXXPtdRZ68ckZ68BPrMU'
api_secret = '0eCJTH1XqC3Wv55Tkg4QBCnD84DZ1cDfPZ3DtJ6MoME2OrJNLubfTJyD8vR4sCk2'
client = Client(api_key, api_secret)

def fetch_historical_data(symbol, interval, start, end, max_retries=5, initial_retry_delay=2):
    def fetch_data_chunk(start_time, end_time):
        try:
            start_str = pd.to_datetime(start_time, unit='ms').strftime('%d %b, %Y %H:%M:%S')
            end_str = pd.to_datetime(end_time, unit='ms').strftime('%d %b, %Y %H:%M:%S')
            klines = client.get_historical_klines(symbol, interval, start_str=start_str, end_str=end_str)
            if not klines:
                logger.warning(f"No data received for {symbol} from {start_str} to {end_str}.")
            return klines
        except Exception as e:
            logger.error(f"Error fetching data chunk: {e}")
            raise

    retries = 0
    retry_delay = initial_retry_delay

    if "ago" in start:
        start_time = datetime.now() - relativedelta(years=3)
    else:
        start_time = pd.to_datetime(start)
    
    if end == "now UTC":
        end_time = datetime.now()
    else:
        end_time = pd.to_datetime(end)
    
    start_time = start_time.timestamp() * 1000
    end_time = end_time.timestamp() * 1000

    while retries < max_retries:
        try:
            all_klines = []
            current_start_time = start_time

            while current_start_time < end_time:
                batch_end_time = min(current_start_time + (1000 * 60 * 60 * 24 * 30), end_time)  # Up to a month at a time
                klines_batch = fetch_data_chunk(current_start_time, batch_end_time)
                all_klines.extend(klines_batch)
                current_start_time = batch_end_time + 1  # Move to the next batch

            if not all_klines:
                logger.error("No data fetched after retries.")
                return None

            data = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                     'taker_buy_quote_asset_volume', 'ignore'])
            data = data.apply(pd.to_numeric, errors='coerce')
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            if data.empty:
                logger.error("Fetched data is empty.")
                return None

            logger.info(f"Data fetched successfully. Data shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            retries += 1
            if retries < max_retries:
                delay = retry_delay * (2 ** retries)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Maximum retries reached. Returning None.")
                return None

# Function to add indicators
def add_indicators(data):
    try:
        required_columns = ['close', 'high', 'low', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing required column: {column}")

        # Adding more indicators
        data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        data['SMA50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['SMA200'] = ta.trend.sma_indicator(data['close'], window=200)
        
        # Normalize RSI: Dividing by 100 ensures it's between 0 and 1
        data['RSI'] = ta.momentum.RSIIndicator(data['close']).rsi() / 100

        # Bollinger Bands
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['close'])
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['close'])

        # Average True Range (ATR)
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

        # On Balance Volume (OBV)
        data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])

        # Trend: Bullish if SMA50 > SMA200, else Bearish
        data['Bullish_Trend'] = (data['SMA50'] > data['SMA200']).astype(int)

        # Stochastic Oscillator
        def stochastic_oscillator(high, low, close, window=14):
            high_roll = high.rolling(window=window).max()
            low_roll = low.rolling(window=window).min()
            return (close - low_roll) / (high_roll - low_roll)
        
        data['Stochastic_Oscillator'] = stochastic_oscillator(data['high'], data['low'], data['close'])

        # Adding EMA indicators
        data['EMA_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['EMA_26'] = ta.trend.ema_indicator(data['close'], window=26)

        # Dropping NA values to ensure clean data for prediction
        data.dropna(inplace=True)

        logger.info("Indicators added successfully.")
        return data
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return data
    
def prepare_data(data):
    try:
        data = data.copy()
        logger.info("Initial Data Shape: %s", data.shape)
        logger.info("Initial Data Head:\n%s", data.head())

        # Ensure all required features are present
        required_features = [
            'SMA50', 'SMA200', 'RSI',
            'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV',
            'Stochastic_Oscillator', 'Bullish_Trend', 'EMA_12', 'EMA_26'
        ]
        missing_columns = [col for col in required_features if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        # Prepare features and target
        X = data[required_features]
        y = data['close'].shift(-1)

        logger.info("Features Shape Before Dropna: %s", X.shape)
        logger.info("Target Shape Before Dropna: %s", y.shape)

        # Align features and target
        X = X.iloc[:-1].reset_index(drop=True)
        y = y.dropna().reset_index(drop=True)
        
        # Make sure X and y have the same length after alignment
        if len(X) != len(y):
            logger.warning("Length mismatch between features and target after alignment. Adjusting...")
            min_len = min(len(X), len(y))
            X = X.head(min_len)
            y = y.head(min_len)

        # Apply StandardScaler to scale the data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        logger.info("Features Shape After Scaling: %s", X_scaled.shape)
        logger.info("Target Shape After Scaling: %s", y.shape)

        # Check if there is enough data for prediction
        if len(X_scaled) < 20:
            logger.warning("Not enough data for prediction. Minimum required data points: 20")
            return None, None

        # Check if scaled features or target data are empty
        if X_scaled.empty or y.empty:
            logger.warning("Features or target data is empty after preparation.")
            return None, None

        return X_scaled, y

    except ValueError as e:
        logger.error(f"ValueError in prepare_data: {e}")
        return None, None
    except KeyError as e:
        logger.error(f"KeyError in prepare_data: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None
    
def optimize_model(X, y):
    def objective(trial):
        try:
            # Suggest parameters for the XGBoost model
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            # Initialize and train the model
            model = XGBRegressor(**param, random_state=42)
            score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
            rmse = (-score.mean()) ** 0.5
            return rmse
        except Exception as e:
            logger.error(f"Error in model optimization trial: {e}")
            return float('inf')  # Return a large value in case of error
    
    # Optimize the model with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, n_jobs=-1)
    # Best hyperparameters
    best_params = study.best_params
    logger.info("Best parameters found: %s", best_params)
    # Train final model with best hyperparameters
    final_model = XGBRegressor(**best_params, random_state=42)
    final_model.fit(X, y)
    # Save the model for reuse
    joblib.dump(final_model, 'best_model.pkl')
    return final_model, best_params

def evaluate_model(X_test, y_test):
    model = joblib.load('best_model.pkl')

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Model Evaluation Metrics:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    return rmse, mae, r2

def test_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate_model(X_test, y_test)


async def send_message(message):
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')  # Use environment variable for the bot token
    chat_id = os.getenv('TELEGRAM_CHAT_ID')  # Use environment variable for the chat ID

    if not bot_token or not chat_id:
        logger.error("Bot token or chat ID not set.")
        return

    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }

    logger.info(f"Payload for Telegram API: {payload}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=payload)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx and 5xx)

            response_json = response.json()
            if not response_json.get('ok', False):
                error_msg = response_json.get('description', 'Unknown error')
                logger.error(f"Telegram API error: {error_msg}")
            else:
                logger.info(f"Message sent to Telegram chat {chat_id}: {message}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error occurred while sending message: {e} - Status code: {e.response.status_code}")
        logger.error(f"Response content: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error occurred while sending message: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while sending message: {e}")
        
async def notify_signal(signal_message):
    await send_message(signal_message)

def calculate_volatility(data):
    """Calculate historical volatility from price data."""
    returns = data['close'].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)  # Annualize volatility
    logger.info(f"Calculated Volatility: {volatility}")
    return volatility


def get_dynamic_threshold_ranges(data):
    """Generate threshold ranges based on historical data analysis."""
    volatility = calculate_volatility(data)

    # Example dynamic adjustment based on volatility
    rsi_low_base = 30
    rsi_high_base = 70
    sentiment_positive_base = 0.1
    sentiment_negative_base = -0.1

    volatility_factor = 1 + (volatility / 100)  # Example adjustment factor

    rsi_thresholds_range = [
        (rsi_low_base * volatility_factor, rsi_high_base * volatility_factor)
        for volatility_factor in [0.9, 1.0, 1.1]
    ]
    sentiment_thresholds_range = [
        (sentiment_positive_base * volatility_factor, sentiment_negative_base * volatility_factor)
        for volatility_factor in [0.9, 1.0, 1.1]
    ]

    return rsi_thresholds_range, sentiment_thresholds_range

def adjust_thresholds(volatility, base_rsi_thresholds=(30, 70)):
    """Adjust RSI thresholds based on historical volatility."""
    if volatility < 0:
        raise ValueError("Volatility must be a non-negative value.")
    
    # Example adjustment factor
    volatility_factor = max(0.5, min(2.0, 1 + (volatility / 100)))  # Adjust factor between 0.5 and 2.0
    
    # Adjust RSI thresholds
    rsi_thresholds = (
        max(0, base_rsi_thresholds[0] * volatility_factor),  # Ensure thresholds are non-negative
        min(100, base_rsi_thresholds[1] * volatility_factor)  # Ensure thresholds do not exceed 100
    )
    
    return rsi_thresholds, None

def determine_trade_signal(predicted_close, current_close, latest_data, base_rsi_thresholds=(30, 70)):
    """Generate trade signals based on price prediction and RSI, without sentiment score."""
    try:
        rsi = latest_data['rsi'].iloc[-1] if 'rsi' in latest_data.columns and not latest_data['rsi'].empty else None
        volatility = calculate_volatility(latest_data) if 'close' in latest_data.columns and not latest_data['close'].empty else 1.0

        rsi_thresholds, _ = adjust_thresholds(volatility, base_rsi_thresholds)

        rsi_low, rsi_high = rsi_thresholds

        if predicted_close > current_close and (rsi is not None and rsi < rsi_high):
            return 1  # Buy
        elif predicted_close < current_close and (rsi is not None and rsi > rsi_low):
            return -1  # Sell
        else:
            return 0  # Hold
    except Exception as e:
        logger.error("Error determining trade signal: %s", e)
        return 0  # Hold as a default fallback


def evaluate_strategy(trade_signals, actual_returns):

    if len(trade_signals) != len(actual_returns):
        raise ValueError("Length of trade_signals and actual_returns must be the same.")

    # Convert trade_signals to numpy array for easier calculations
    trade_signals = np.array(trade_signals)
    actual_returns = np.array(actual_returns)

    # Calculate strategy returns
    strategy_returns = trade_signals * actual_returns

    # Calculate total return and annualized return
    total_return = strategy_returns.sum()
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1  # Example annualization

    # Calculate Sharpe ratio
    excess_returns = strategy_returns - actual_returns.mean()  # Excess returns over average market returns
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualize Sharpe ratio

    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peak - cumulative_returns) / peak
    max_drawdown = drawdowns.max()

    # Calculate hit rate
    num_signals = len(trade_signals)
    correct_signals = np.sum(
        (trade_signals == 1) & (actual_returns > 0) | 
        (trade_signals == -1) & (actual_returns < 0)
    )
    hit_rate = correct_signals / num_signals if num_signals > 0 else 0

    # Return performance metrics
    return {
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate
    }

def sensitivity_analysis(data, model):
    rsi_thresholds_range, _ = get_dynamic_threshold_ranges(data)
    
    best_score = -float('inf')
    best_thresholds = None

    for rsi_thresholds in rsi_thresholds_range:
        trade_signals = []
        for row in data.itertuples():
            signal = determine_trade_signal(
                row.predicted_close,
                row.current_close,
                data,
                rsi_thresholds=rsi_thresholds
            )
            trade_signals.append(signal)

        # Evaluate performance with current thresholds
        score = evaluate_strategy(trade_signals, data['actual_returns'])
        if score > best_score:
            best_score = score
            best_thresholds = rsi_thresholds

    return best_thresholds


async def real_time_trading(symbol, model, X_train, y_train, analyzer, base_rsi_thresholds=(30, 70)):
    try:
        logger.info(f"Starting real-time trading for {symbol}")

        end_time = "now UTC"
        start_time = (datetime.now() - relativedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

        # Fetch historical data
        try:
            data_1h = fetch_historical_data(symbol, Client.KLINE_INTERVAL_1HOUR, start_time, end_time)
            data_d = fetch_historical_data(symbol, Client.KLINE_INTERVAL_1DAY, start_time, end_time)
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return 0, "Error fetching historical data."

        if data_1h is not None and len(data_1h) > 50:
            logger.info(f"Fetched {len(data_1h)} 1-hour data points for {symbol}")

            try:
                data_1h = add_indicators(data_1h)
                data_d = add_indicators(data_d)
            except Exception as e:
                logger.error(f"Error adding indicators: {e}")
                return 0, "Error adding indicators."

            try:
                sentiment_score = await fetch_news_sentiment(symbol, analyzer)
                logger.info(f"Sentiment score for {symbol}: {sentiment_score}")
            except Exception as e:
                logger.error(f"Error fetching sentiment score: {e}")
                sentiment_score = 0  # Default or neutral sentiment score for logging only

            try:
                X_latest, _ = prepare_data(data_1h)
                if X_latest is not None and not X_latest.empty:
                    X_latest = X_latest.iloc[-1:].values
                    predicted_close = model.predict(X_latest)[0]
                    current_close = data_1h['close'].iloc[-1]
                    logger.info(f"{symbol} - Predicted close: {predicted_close}, Current close: {current_close}")

                    potential_profit = (predicted_close - current_close) * 1  # Dummy quantity
                    profit_or_loss_percentage = (potential_profit / current_close) * 100

                    try:
                        best_thresholds = sensitivity_analysis(data_1h, model)
                        if best_thresholds:
                            rsi_thresholds = best_thresholds
                        else:
                            rsi_thresholds = base_rsi_thresholds
                    except Exception as e:
                        logger.error(f"Error during sensitivity analysis: {e}")
                        rsi_thresholds = base_rsi_thresholds

                    # Ensure `determine_trade_signal_enhanced` is the correct function name and it accepts the correct parameters
                    trade_signal = determine_trade_signal_enhanced(
                        predicted_close, 
                        current_close, 
                        data_1h,
                        rsi_thresholds=rsi_thresholds
                    )

                    if trade_signal == 1:
                        message = (
                            f"📈 **Buy Signal for {symbol}**\n\n"
                            f"💹 **Predicted Price**: ${predicted_close:.2f}\n"
                            f"💰 **Current Price**: ${current_close:.2f}\n\n"
                            f"💡 **Recommendation**: Price increase predicted. Consider buying.\n"
                            f"🛒 **Potential Gain**: ${potential_profit:.2f} ({profit_or_loss_percentage:.2f}%)"
                        )
                    elif trade_signal == -1:
                        message = (
                            f"🚨 **Sell Signal for {symbol}**\n\n"
                            f"📉 **Predicted Price**: ${predicted_close:.2f}\n"
                            f"💰 **Current Price**: ${current_close:.2f}\n\n"
                            f"💡 **Recommendation**: Price drop predicted. Consider selling.\n"
                            f"📉 **Potential Loss Mitigation**: ${-potential_profit:.2f} ({profit_or_loss_percentage:.2f}%)"
                        )
                    else:
                        message = (
                            f"⚠️ **No Strong Signal for {symbol}**\n\n"
                            f"🔮 **Predicted Price**: ${predicted_close:.2f}\n"
                            f"💰 **Current Price**: ${current_close:.2f}\n\n"
                            f"🔍 **Analysis**: No significant change predicted. Hold off on any trading action for now."
                        )

                    await send_message(message)
                    return trade_signal, message

                else:
                    logger.error(f"Prepared data is empty for {symbol}")
                    return 0, "Insufficient data for prediction."

            except Exception as e:
                logger.error(f"Error preparing data for prediction: {e}")
                return 0, "Error preparing data for prediction."

        else:
            logger.error(f"Insufficient data for {symbol}")
            return 0, "No recent data available."

    except Exception as e:
        logger.error(f"Error during real-time trading: {e}")
        return 0, f"Error: {e}"


# Constants
HOURS_OF_DATA = 24 * 365  # Example: 24 hours/day * 365 days/year
#HOURS_OF_DATA = 24 * 30  # Example: 24 hours/day * 30 days
# Define hours_of_data as a timedelta object representing a certain number of hours
#hours_of_data = timedelta(hours=5)  

# Define your value for HOURS_OF_DATA
#VALID_SYMBOLS = set(['NEARUSDT', 'TONUSDT', 'ETHUSDT', 'FETUSDT', 'CKBUSDT', 'XRPUSDT', 'SOLUSDT', 'WEETHUSDT', 'HBARUSDT', 'DOGSUSDT', 'AVAXUSDT', 'USDUSDT', 'WSTETHUSDT', 'USDEUSDT', 'IMXUSDT', 'XLMUSDT', 'TRXUSDT', 'AAVEUSDT', 'DOGEUSDT', 'WIFUSDT'])

VALID_SYMBOLS = set(['ETHUSDT' ])


async def process_coin(symbol):
    if symbol not in VALID_SYMBOLS:
        logger.error("Invalid symbol %s", symbol)
        return

    end_time = "now UTC"
    start_time = (datetime.now() - timedelta(hours=HOURS_OF_DATA)).strftime('%Y-%m-%d %H:%M:%S')

    try:
        try:
            historical_data_1h = fetch_historical_data(symbol, '1h', start_time, end_time)
            historical_data_d = fetch_historical_data(symbol, '1d', start_time, end_time)
        except Exception as e:
            logger.error("Error fetching historical data for %s: %s", symbol, e)
            return

        if isinstance(historical_data_1h, pd.DataFrame) and not historical_data_1h.empty and len(historical_data_1h) > 20:
            logger.info("1-hour historical data fetched for %s, adding indicators...", symbol)
            historical_data_1h = add_indicators(historical_data_1h)
            logger.info("Technical indicators added for %s.", symbol)

            X_train, y_train = prepare_data(historical_data_1h)

            if X_train is not None and y_train is not None and len(X_train) > 20:
                logger.info("Data prepared for model training for %s. Optimizing model...", symbol)
                model, best_params = optimize_model(X_train, y_train)
                logger.info("Model optimized with parameters: %s", best_params)
                
                logger.info("Performing real-time trading for %s...", symbol)
                try:
                    
                    trade_signal, message = await real_time_trading(symbol, model, X_train, y_train, analyzer)
                    logger.info("Trade signal for %s: %d", symbol, trade_signal)
                except Exception as e:
                    logger.error("Error during real-time trading for %s: %s", symbol, e)
                    return
                
                try:
                    await notify_signal(message)
                    logger.info("Notification sent for %s.", symbol)
                except Exception as e:
                    logger.error("Error sending notification for %s: %s", symbol, e)
            else:
                logger.error("Insufficient data after preparation for %s.", symbol)
        else:
            logger.error("Insufficient historical data for %s.", symbol)
    except Exception as e:
        logger.error("Error processing coin %s: %s", symbol, e)

async def main():
    #coins = ['NEARUSDT', 'TONUSDT', 'ETHUSDT', 'FETUSDT', 'CKBUSDT', 'XRPUSDT', 'SOLUSDT', 'WEETHUSDT', 'HBARUSDT', 'DOGSUSDT', 'AVAXUSDT', 'USDUSDT', 'WSTETHUSDT', 'USDEUSDT', 'IMXUSDT', 'XLMUSDT', 'TRXUSDT', 'AAVEUSDT', 'DOGEUSDT', 'WIFUSDT']  # Example symbols
    coins = ['ETHUSDT']
    tasks = [process_coin(symbol) for symbol in coins]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import ta
from binance.client import Client
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
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


# Initialize Binance client
api_key = 'cLVgIkkwrnKDoOJtBzEUVo2fP7a09fVhFLDVl1A4j6LosXXPtdRZ68ckZ68BPrMU'
api_secret = '0eCJTH1XqC3Wv55Tkg4QBCnD84DZ1cDfPZ3DtJ6MoME2OrJNLubfTJyD8vR4sCk2'
client = Client(api_key, api_secret)

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()
    # Ensure 'volume' column is positive
    df = df[df['volume'] > 0]
    # Add more data cleaning steps as needed
    return df

def fetch_historical_data(symbol, interval, start, end, max_retries=5, initial_retry_delay=2):
    def fetch_data_chunk(start_str, end_str):
        try:
            klines = client.get_historical_klines(symbol, interval, start_str=start_str, end_str=end_str)
            if not klines or len(klines) == 0:
                logger.warning(f"No data received for {symbol} from {start_str} to {end_str}.")
                return []
            return klines
        except Exception as e:
            logger.error(f"Error fetching data chunk from {start_str} to {end_str}: {e}")
            raise

    retries = 0
    retry_delay = initial_retry_delay

    try:
        # Convert start and end to datetime
        if "ago" in start:
            start_time = datetime.now() - relativedelta(years=2)
        else:
            start_time = pd.to_datetime(start)
        
        if end == "now UTC":
            end_time = datetime.now()
        else:
            end_time = pd.to_datetime(end)

        current_start_time = start_time
        all_klines = []

        # Main loop for fetching data
        while retries < max_retries:
            try:
                while current_start_time < end_time:
                    batch_end_time = min(current_start_time + relativedelta(months=1), end_time)
                    current_start_str = current_start_time.strftime('%d %b, %Y %H:%M:%S')
                    batch_end_str = batch_end_time.strftime('%d %b, %Y %H:%M:%S')

                    klines_batch = fetch_data_chunk(current_start_str, batch_end_str)
                    if klines_batch:  # Add non-empty batch to all_klines
                        all_klines.extend(klines_batch)
                    else:
                        logger.warning(f"No data fetched for {symbol} from {current_start_str} to {batch_end_str}.")

                    # If we reach the end of the requested time range, stop
                    if batch_end_time >= end_time:
                        break
                    
                    # Move to next batch
                    current_start_time = batch_end_time + timedelta(milliseconds=1)

                # After loop check if we have any data
                if not all_klines:
                    logger.error("No data was fetched after multiple attempts.")
                    return None

                # Define columns for the DataFrame
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                           'close_time', 'quote_asset_volume', 'number_of_trades', 
                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                           'ignore']

                # Create DataFrame
                data = pd.DataFrame(all_klines, columns=columns)

                # Convert to appropriate types and set timestamp as index
                data = data.apply(pd.to_numeric, errors='coerce')
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('timestamp', inplace=True)

                # Preprocess data (ensure the data is valid after preprocessing)
                data = preprocess_data(data)
                if data is None or data.empty:
                    logger.error("Data is empty after preprocessing.")
                    return None

                # Add indicators (handle None and empty cases properly)
                data = add_indicators(data)
                if data is None or data.empty:
                    logger.error("Data is empty after adding indicators.")
                    return None

                # Calculate actual returns if it's missing
                if 'actual_returns' not in data.columns:
                    data['actual_returns'] = data['close'].pct_change().shift(-1)
                    if data['actual_returns'].isnull().all():
                        logger.error("Failed to calculate actual returns. Data is invalid.")
                        return None

                # Prepare data for the model (e.g., features and target)
                X, y = prepare_data(data)
                if X is None or y is None:
                    logger.error("Prepared data is empty.")
                    return None

                logger.info(f"Data fetched successfully. Shape: {data.shape}")
                return data

            except Exception as e:
                retries += 1
                logger.error(f"Error during data fetch: {e}. Retry {retries} of {max_retries}.")
                if retries < max_retries:
                    delay = retry_delay * (2 ** retries)  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("Max retries reached. No data fetched.")
                    return None
    except Exception as e:
        logger.error(f"Critical error in fetch_historical_data: {e}")
        return None

# Function to add indicators
def add_indicators(data):
    if data is None or data.empty:
        logger.error("Data is None or empty when adding indicators.")
        return None
    try:
        required_columns = ['close', 'high', 'low', 'volume']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Missing required column: {column}")

        # Adding indicators
        data['SMA20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['SMA50'] = ta.trend.sma_indicator(data['close'], window=50)

        # RSI
        data['RSI'] = ta.momentum.RSIIndicator(data['close']).rsi()

        # Bollinger Bands
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['close'], window=20)
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['close'], window=20)

        # On Balance Volume (OBV)
        data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])

        # Average True Range (ATR)
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

        # Moving Average Convergence Divergence (MACD)
        data['MACD'] = ta.trend.macd(data['close'])
        data['MACD_Signal'] = ta.trend.macd_signal(data['close'])
        data['MACD_Histogram'] = ta.trend.macd_diff(data['close'])

        # Fill NA values instead of dropping rows
        data.fillna(method='ffill', inplace=True)  # Forward fill to avoid empty data

        # Optional: Check if data is still empty
        if data.empty:
            logger.error("Data is empty after adding indicators.")
            return None

        logger.info("Indicators added successfully.")
        return data
    except Exception as e:
        logger.error(f"Error in add_indicators: {e}")
        return None

def prepare_data(data):
    try:
        data = data.copy()
        logger.info("Initial Data Shape: %s", data.shape)
        logger.info("Initial Data Head:\n%s", data.head())

        # Ensure all required features are present
        required_features = ['SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_Histogram', 'Bollinger_High', 'Bollinger_Low', 'ATR', 'OBV', 'Stochastic_Oscillator']
        missing_columns = [col for col in required_features if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {', '.join(missing_columns)}")
            # Optionally, drop missing columns from required features and proceed
            required_features = [col for col in required_features if col in data.columns]

        # Prepare features and target
        X = data[required_features]
        y = data['close'].shift(-1)

        logger.info("Features Shape Before Dropna: %s", X.shape)
        logger.info("Target Shape Before Dropna: %s", y.shape)

        # Align features and target
        X = X[:-1]
        y = y.dropna()
        X = X.loc[y.index]

        logger.info("Features Shape After Dropna: %s", X.shape)
        logger.info("Target Shape After Dropna: %s", y.shape)

        # Check if there is enough data for prediction
        if len(X) < 20:
            logger.warning("Not enough data for prediction. Minimum required data points: 20")
            return None, None

        if X.empty or y.empty:
            logger.warning("Features or target data is empty after preparation.")
            return None, None

        return X, y

    except ValueError as e:
        logger.error(f"ValueError in prepare_data: {e}")
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

    try:
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        logger.info("Model Evaluation Metrics:")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        return rmse, mae, r2
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        return None, None, None

def test_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        evaluate_model(X_test, y_test)
    except Exception as e:
        logger.error(f"Error during model testing: {e}")


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

def adjust_thresholds(volatility, base_rsi_thresholds=(30, 70)):
    """Adjust RSI thresholds based on historical volatility."""
    if volatility < 0:
        raise ValueError("Volatility must be a non-negative value.")
    
    volatility_factor = max(0.5, min(2.0, 1 + (volatility / 100)))  # Adjust factor between 0.5 and 2.0
    
    rsi_thresholds = (
        max(0, base_rsi_thresholds[0] * volatility_factor),  # Ensure thresholds are non-negative
        min(100, base_rsi_thresholds[1] * volatility_factor)  # Ensure thresholds do not exceed 100
    )
    
    return rsi_thresholds

def get_dynamic_threshold_ranges(data):
    """Generate threshold ranges based on historical data analysis."""
    if data.empty:
        logger.error("Input data is empty. Cannot calculate thresholds.")
        return [(30, 70)]  # Return default thresholds if data is empty

    volatility = calculate_volatility(data)

    if pd.isna(volatility) or volatility < 0:
        logger.error("Calculated volatility is invalid. Setting default thresholds.")
        volatility = 0

    rsi_low_base = 30
    rsi_high_base = 70
    volatility_factor = 1 + (volatility / 100)

    rsi_thresholds_range = [
        (rsi_low_base * volatility_factor, rsi_high_base * volatility_factor)
        for volatility_factor in [0.9, 1.0, 1.1]
    ]

    return rsi_thresholds_range

def determine_trade_signal(predicted_close, current_close, latest_data, base_rsi_thresholds=(30, 70), ma_window=20):
    """Generate trade signals based on price prediction, RSI, and moving average."""
    try:
        logger.info(f"Predicted Close: {predicted_close}, Current Close: {current_close}")

        # Convert to float to ensure valid numeric operations
        predicted_close = float(predicted_close)
        current_close = float(current_close)

        rsi = latest_data['RSI'].iloc[-1] if 'RSI' in latest_data.columns and not latest_data['RSI'].empty else None
        volatility = calculate_volatility(latest_data) if 'close' in latest_data.columns and not latest_data['close'].empty else 1.0

        rsi_thresholds = adjust_thresholds(volatility, base_rsi_thresholds) if volatility > 0 else base_rsi_thresholds
        rsi_low, rsi_high = rsi_thresholds

        moving_average = (
            latest_data['close'].rolling(window=ma_window).mean().iloc[-1]
            if 'close' in latest_data.columns and not latest_data['close'].empty else current_close
        )

        signal_strength = predicted_close - current_close
        relative_difference = abs(signal_strength) / current_close if current_close != 0 else 0

        if predicted_close > current_close:
            if (rsi is not None and rsi < rsi_high) or relative_difference > 0.02:
                if current_close > moving_average or signal_strength > 0.005 * volatility:
                    logger.info("Generated Buy signal.")
                    return 1

        elif predicted_close < current_close:
            if (rsi is not None and rsi > rsi_low) or relative_difference > 0.02:
                if current_close < moving_average or signal_strength < -0.005 * volatility:
                    logger.info("Generated Sell signal.")
                    return -1

        if relative_difference > 0.03:
            logger.info("Fallback signal based on relative difference.")
            return 1 if predicted_close > current_close else -1

        logger.info("No clear signal generated, holding position.")
        return 0

    except Exception as e:
        logger.error("Error determining trade signal: %s", e)
        return 0

def evaluate_strategy(trade_signals, actual_returns):
    """Evaluate trading strategy performance metrics."""
    if len(trade_signals) != len(actual_returns):
        raise ValueError("Length of trade_signals and actual_returns must be the same.")

    trade_signals = np.array(trade_signals)
    actual_returns = np.array(actual_returns)

    strategy_returns = trade_signals * actual_returns
    total_return = strategy_returns.sum()
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    excess_returns = strategy_returns - actual_returns.mean()
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peak - cumulative_returns) / peak
    max_drawdown = drawdowns.max()
    num_signals = len(trade_signals)
    correct_signals = np.sum(
        (trade_signals == 1) & (actual_returns > 0) |
        (trade_signals == -1) & (actual_returns < 0)
    )
    hit_rate = correct_signals / num_signals if num_signals > 0 else 0

    return {
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate
    }

def sensitivity_analysis(data, model):
    """Perform sensitivity analysis to find the best thresholds for the trading strategy."""
    try:
        # Ensure data is not empty
        if data.empty:
            logger.error("Input data is empty.")
            return None

        # Ensure that dynamic thresholds are properly fetched
        rsi_thresholds_range = get_dynamic_threshold_ranges(data)

        best_score = -float('inf')
        best_thresholds = (None, None)

        # Iterate over all combinations of RSI thresholds
        for rsi_thresholds in rsi_thresholds_range:
            trade_signals = []
            for row in data.itertuples(index=False):
                predicted_close = getattr(row, 'predicted_close', None)
                current_close = getattr(row, 'current_close', None)

                # Log the values and their types
                logger.info(f"Predicted Close: {predicted_close} (Type: {type(predicted_close)}), Current Close: {current_close} (Type: {type(current_close)})")

                # Check if the values are valid
                if isinstance(predicted_close, (int, float)) and isinstance(current_close, (int, float)):
                    if pd.isna(predicted_close) or pd.isna(current_close):
                        logger.error("Predicted close or current close is NaN.")
                        continue  # Skip this iteration if values are invalid

                    signal = determine_trade_signal(
                        predicted_close,
                        current_close,
                        data,
                        base_rsi_thresholds=rsi_thresholds
                    )
                    trade_signals.append(signal)
                else:
                    logger.error("Invalid input types for predicted_close or current_close.")

            # Ensure 'actual_returns' exists before evaluating strategy
            if 'actual_returns' in data.columns:
                score = evaluate_strategy(trade_signals, data['actual_returns'])
                if score['annualized_return'] > best_score:
                    best_score = score['annualized_return']
                    best_thresholds = rsi_thresholds  # Use just the thresholds
            else:
                logger.error("actual_returns column is missing in the data.")

        return best_thresholds

    except Exception as e:
        logger.error("Error during sensitivity analysis: %s", e)
        return None
    
async def real_time_trading(symbol, model_path, X_train, y_train, analyzer, base_rsi_thresholds=(30, 70)):
    try:
        logger.info(f"Starting real-time trading for {symbol}")

        end_time = "now UTC"
        start_time = (datetime.now() - relativedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')

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
                X_latest, _ = prepare_data(data_1h)
                if X_latest is not None and not X_latest.empty:
                    X_latest = X_latest.iloc[-1:].values
                    logger.info(f"X_latest: {X_latest}")

                    try:
                        model = joblib.load('best_model.pkl')
                    except Exception as e:
                        logger.error(f"Error loading model: {e}")
                        return 0, "Error loading model."

                    try:
                        predicted_close = model.predict(X_latest)[0]
                    except Exception as e:
                        logger.error(f"Error predicting close: {e}")
                        predicted_close = None

                    if data_1h is not None and not data_1h.empty:
                        try:
                            current_close = data_1h['close'].iloc[-1]
                        except Exception as e:
                            logger.error(f"Error fetching current close: {e}")
                            current_close = None
                    else:
                        logger.error("data_1h is empty. Cannot fetch current close.")
                        return 0, "No recent data available."

                    if predicted_close is None or current_close is None:
                        logger.error("Predicted close or current close is None.")
                        return 0, "Prediction or current close is None."

                    logger.info(f"{symbol} - Predicted close: {predicted_close}, Current close: {current_close}")

                    potential_profit = (predicted_close - current_close) * 1
                    profit_or_loss_percentage = (potential_profit / current_close) * 100

                    try:
                        best_thresholds = sensitivity_analysis(data_1h, model)
                        rsi_thresholds = best_thresholds if best_thresholds else base_rsi_thresholds
                    except Exception as e:
                        logger.error(f"Error during sensitivity analysis: {e}")
                        rsi_thresholds = base_rsi_thresholds

                    try:
                        trade_signal = determine_trade_signal(
                            predicted_close=predicted_close,
                            current_close=current_close,
                            latest_data=data_1h,
                            base_rsi_thresholds=rsi_thresholds
                        )
                    except Exception as e:
                        logger.error(f"Error determining trade signal: {e}")
                        trade_signal = 0

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
# VALID_SYMBOLS = set([
#     'BTCUSDT',  # Bitcoin
#     'ETHUSDT',  # Ethereum
#     'BNBUSDT',  # Binance Coin
#     'SOLUSDT',  # Solana
#     'ADAUSDT',  # Cardano
#     'XRPUSDT',  # Ripple
#     'DOTUSDT',  # Polkadot
#     'LINKUSDT', # Chainlink
#     'DOGEUSDT', # Dogecoin
#     'LTCUSDT',  # Litecoin
#     'UNIUSDT',  # Uniswap
#     'AVAXUSDT', # Avalanche
#     'MATICUSDT',# Polygon
#     'SHIBUSDT', # Shiba Inu
#     'ATOMUSDT', # Cosmos
#     'BTTUSDT',  # BitTorrent
#     'FILUSDT',  # Filecoin
#     'ICPUSDT',  # Internet Computer
#     'SANDUSDT', # The Sandbox
#     'ALGOUSDT'  # Algorand
# ])
VALID_SYMBOLS = set(['BTCUSDT' ])


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
    
    # coins = [
    #     'BTCUSDT',  # Bitcoin
    #     'ETHUSDT',  # Ethereum
    #     'BNBUSDT',  # Binance Coin
    #     'SOLUSDT',  # Solana
    #     'ADAUSDT',  # Cardano
    #     'XRPUSDT',  # Ripple
    #     'DOTUSDT',  # Polkadot
    #     'LINKUSDT', # Chainlink
    #     'DOGEUSDT', # Dogecoin
    #     'LTCUSDT',  # Litecoin
    #     'UNIUSDT',  # Uniswap
    #     'AVAXUSDT', # Avalanche
    #     'MATICUSDT',# Polygon
    #     'SHIBUSDT', # Shiba Inu
    #     'ATOMUSDT', # Cosmos
    #     'BTTUSDT',  # BitTorrent
    #     'FILUSDT',  # Filecoin
    #     'ICPUSDT',  # Internet Computer
    #     'SANDUSDT', # The Sandbox
    #     'ALGOUSDT'  # Algorand
    # ]
    coins = ['BTCUSDT']
    tasks = [process_coin(symbol) for symbol in coins]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

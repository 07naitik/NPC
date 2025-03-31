import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy.stats import norm
import sys
import argparse
import traceback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy.stats import norm
import sys
import argparse
import traceback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import psycopg2
from psycopg2.extras import RealDictCursor

TRADING_HOURS_PER_YEAR = 24 * 365  # For crypto, which trades 24/7

# Try to import dotenv for loading environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from .env file
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. If using .env file, install with: pip install python-dotenv")

# Print available environment variables for Timescale
timescale_vars = [var for var in os.environ if var.startswith("TIMESCALE_")]
if timescale_vars:
    print(f"Found Timescale environment variables: {', '.join(timescale_vars)}")
else:
    print("WARNING: No Timescale environment variables found")

# Try to import Streamlit, but don't fail if it's not installed
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Install with: pip install streamlit")

# Constants for volatility calculation
TRADING_HOURS_PER_YEAR = 24 * 365  # For crypto, which trades 24/7

# Lookback periods for volatility calculation (in hours)
lookback_periods = {
    "2 weeks": 24 * 14,  # 336 hours
    "1 week": 24 * 7,    # 168 hours
    "3 days": 24 * 3,    # 72 hours
    "1 day": 24,         # 24 hours
    "12 hours": 12,      # 12 hours 
    "6 hours": 6,        # 6 hours
}

def setup_timescale_connection():
    """
    Set up the Timescale database connection.
    
    Returns:
        connection: Database connection object
    """
    # Get database credentials from environment variables
    db_host = os.getenv("TIMESCALE_HOST")
    db_port = os.getenv("TIMESCALE_PORT", "34569")  # Using custom port 34569 instead of default 5432
    db_name = os.getenv("TIMESCALE_DATABASE")
    db_user = os.getenv("TIMESCALE_USER")
    db_password = os.getenv("TIMESCALE_PASSWORD")
    
    # Check if essential environment variables are missing
    missing_vars = []
    if not db_host:
        missing_vars.append("TIMESCALE_HOST")
    if not db_name:
        missing_vars.append("TIMESCALE_DATABASE")
    if not db_user:
        missing_vars.append("TIMESCALE_USER")
    if not db_password:
        missing_vars.append("TIMESCALE_PASSWORD")
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set the following environment variables:")
        print("  TIMESCALE_HOST - The hostname or IP of your Timescale server")
        print(f"  TIMESCALE_PORT - The port of your Timescale server (default: {db_port})")
        print("  TIMESCALE_DATABASE - The database name")
        print("  TIMESCALE_USER - The database username")
        print("  TIMESCALE_PASSWORD - The database password")
        print("\nYou can set these in a .env file or directly in your environment.")
        return None
    
    # Print environment variables for debugging (hide sensitive info)
    print(f"Connecting to Timescale DB at {db_host}:{db_port}/{db_name} as {db_user}")
    
    try:
        # Connect to the database with SSL enabled and a connection timeout
        connection = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            sslmode='require',  # Enable SSL
            connect_timeout=10  # 10-second timeout
        )
        print("Successfully connected to Timescale database")
        return connection
    except psycopg2.OperationalError as e:
        print(f"Error connecting to Timescale database: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your IP is allowed in the Timescale firewall/security group")
        print("2. Verify if a VPN connection is required")
        print("3. Check if the database server is running and accessible")
        print("4. Verify the correct port number (34569 in your example, not the default 5432)")
        print(f"5. Try connecting using psql to test direct connectivity:")
        print(f"   psql -h {db_host} -p {db_port} -U {db_user} -d {db_name} sslmode=require")
        return None
    except Exception as e:
        print(f"Unexpected error connecting to Timescale database: {str(e)}")
        return None

# Initialize database connection
db_connection = None

# Define the symbols and trading pairs to analyze
# Individual symbols needed
symbols = [
    "SOL", "USDC", "BTC", "JLP", "JTO", "JUP", "RAY", "FARTCOIN", 
    "ARC", "ETH", "SUI", "DEEP", "PENDLE", "ZRO"
]

# Define trading pairs and how to compute them
trading_pairs_config = [
    {"name": "SOL/USDC", "numerator": "SOL", "denominator": "USDC"},
    {"name": "SOL/BTC", "numerator": "SOL", "denominator": "BTC"},
    {"name": "JLP/SOL", "numerator": "JLP", "denominator": "SOL"},
    {"name": "JTO/SOL", "numerator": "JTO", "denominator": "SOL"},
    {"name": "JUP/SOL", "numerator": "JUP", "denominator": "SOL"},
    {"name": "RAY/SOL", "numerator": "RAY", "denominator": "SOL"},
    {"name": "FARTCOIN/SOL", "numerator": "FARTCOIN", "denominator": "SOL"},
    {"name": "ARC/SOL", "numerator": "ARC", "denominator": "SOL"},
    {"name": "SOL/ETH", "numerator": "SOL", "denominator": "ETH"},
    {"name": "SUI/USDC", "numerator": "SUI", "denominator": "USDC"},
    {"name": "DEEP/SUI", "numerator": "DEEP", "denominator": "SUI"},
    {"name": "SUI/BTC", "numerator": "SUI", "denominator": "BTC"},
    {"name": "ETH/USDC", "numerator": "ETH", "denominator": "USDC"},
    {"name": "PENDLE/ETH", "numerator": "PENDLE", "denominator": "ETH"},
    {"name": "ZRO/ETH", "numerator": "ZRO", "denominator": "ETH"},
    {"name": "ETH/BTC", "numerator": "ETH", "denominator": "BTC"}
]

# Get all unique trading pair names
trading_pairs = [config["name"] for config in trading_pairs_config]

def fetch_symbol_data(symbol, months=3, connection=None):
    """
    Fetch price history for a single symbol from Timescale DB.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., "SOL")
        months (int): Number of months of historical data to retrieve
        connection: The database connection to use
        
    Returns:
        pandas.DataFrame: Dataframe with datetime and close price
    """
    # Make sure we have a database connection
    if connection is None:
        global db_connection
        if db_connection is None:
            db_connection = setup_timescale_connection()
        connection = db_connection
    
    if connection is None:
        print(f"Cannot fetch data for {symbol}: No database connection")
        return None
    
    # Calculate the start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    
    print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # First, let's check the table structure to get the column names
        cursor = connection.cursor()
        
        try:
            # Get table columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'all_crypto_prices'
            """)
            columns = [row[0] for row in cursor.fetchall()]
            print(f"Available columns in public.all_crypto_prices: {columns}")
            
            # Close cursor and create a new one to avoid any transaction issues
            cursor.close()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Identify the correct column names for datetime and close
            date_column = next((col for col in columns if 'time' in col.lower() or 'date' in col.lower()), None)
            close_column = next((col for col in columns if 'close' in col.lower()), None)
            
            if not date_column or not close_column:
                print(f"Could not identify required columns. Available columns: {columns}")
                cursor.close()
                return None
                
            # Query the database for the price history
            query = f"""
                SELECT {date_column} as datetime, {close_column} as close
                FROM public.all_crypto_prices
                WHERE symbol = %s
                AND {date_column} >= %s
                AND {date_column} <= %s
                ORDER BY {date_column}
            """
            
            print(f"Executing query with date_column={date_column}, close_column={close_column}")
            cursor.execute(query, (symbol, start_date.isoformat(), end_date.isoformat()))
            
            # Fetch all results
            records = cursor.fetchall()
            
            # Close the cursor
            cursor.close()
            
            # Print raw response for debugging
            print(f"Raw response data preview for {symbol}:")
            if records:
                print(f"First 2 records: {records[:2]}")
                print(f"Last 2 records: {records[-2:]}")
                print(f"Total records: {len(records)}")
            else:
                print("No data returned")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(records)
            
            if df.empty:
                print(f"No data found for symbol {symbol}")
                return None
                
            # Convert datetime to proper format and set as index
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Keep only unique datetimes by taking the last value for each
            df = df.drop_duplicates(subset=['datetime'], keep='last')
            df = df.set_index('datetime')
            
            # Sort by datetime to ensure data is in chronological order
            df = df.sort_index()
            
            # Check for sufficient data
            if len(df) < 10:  # Arbitrary threshold
                print(f"Insufficient data for symbol {symbol}: only {len(df)} data points found")
                return None
            
            print(f"Retrieved {len(df)} data points for {symbol}")
            return df
            
        except Exception as inner_e:
            print(f"Error in data processing for {symbol}: {str(inner_e)}")
            cursor.close()
            connection.rollback()  # Reset any failed transaction
            
            # Try a simpler approach - get all columns without filtering
            try:
                cursor = connection.cursor(cursor_factory=RealDictCursor)
                print(f"Attempting fallback query for {symbol}...")
                cursor.execute("""
                    SELECT * FROM public.all_crypto_prices 
                    WHERE symbol = %s LIMIT 1
                """, (symbol,))
                
                sample_record = cursor.fetchone()
                cursor.close()
                
                if sample_record:
                    print(f"Sample record columns: {list(sample_record.keys())}")
                    print(f"Sample record: {sample_record}")
                    
                    # Re-attempt with exact column names
                    connection.rollback()
                    cursor = connection.cursor(cursor_factory=RealDictCursor)
                    
                    # Use the actual column names from the sample record
                    date_key = next((k for k in sample_record.keys() if 'time' in k.lower() or 'date' in k.lower()), None)
                    close_key = next((k for k in sample_record.keys() if 'close' in k.lower() or 'price' in k.lower()), None)
                    
                    if date_key and close_key:
                        print(f"Using columns: datetime={date_key}, close={close_key}")
                        query = f"""
                            SELECT {date_key} as datetime, {close_key} as close
                            FROM public.all_crypto_prices
                            WHERE symbol = %s
                            ORDER BY {date_key}
                        """
                        cursor.execute(query, (symbol,))
                        records = cursor.fetchall()
                        cursor.close()
                        
                        df = pd.DataFrame(records)
                        
                        if not df.empty:
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df = df.drop_duplicates(subset=['datetime'], keep='last')
                            df = df.set_index('datetime')
                            df = df.sort_index()
                            
                            print(f"Retrieved {len(df)} data points for {symbol} using fallback approach")
                            return df
                else:
                    print(f"No sample record found for symbol {symbol}")
            except Exception as fallback_e:
                print(f"Fallback query also failed: {str(fallback_e)}")
                cursor.close()
                connection.rollback()
            
            return None
    
    except Exception as e:
        print(f"Error fetching data for symbol {symbol}: {str(e)}")
        traceback.print_exc()
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        connection.rollback()  # Reset the failed transaction
        return None

def fetch_all_symbols_data(symbols, months=3):
    """
    Fetch price history for all symbols.
    
    Args:
        symbols (list): List of cryptocurrency symbols
        months (int): Number of months of historical data to retrieve
        
    Returns:
        dict: Dictionary mapping symbols to their price dataframes
    """
    # First, ensure we have a database connection
    global db_connection
    if db_connection is None:
        db_connection = setup_timescale_connection()
        
    if db_connection is None:
        print("Cannot fetch data: No database connection")
        return {}
    
    symbol_data = {}
    
    for symbol in symbols:
        try:
            symbol_data[symbol] = fetch_symbol_data(symbol, months, db_connection)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            traceback.print_exc()
    
    # Log how many symbols we successfully retrieved
    successful_symbols = sum(1 for df in symbol_data.values() if df is not None)
    print(f"Successfully retrieved data for {successful_symbols}/{len(symbols)} symbols")
    
    return symbol_data

def calculate_pair_price(symbol_data, numerator_symbol, denominator_symbol):
    """
    Calculate price for a trading pair from individual symbol prices.
    Preserves hourly data granularity.
    
    Args:
        symbol_data (dict): Dictionary mapping symbols to their price dataframes
        numerator_symbol (str): Symbol for the numerator (e.g., "SOL")
        denominator_symbol (str): Symbol for the denominator (e.g., "USDC")
        
    Returns:
        pandas.DataFrame: Dataframe with calculated pair price or None if data is missing
    """
    # Get dataframes for both symbols
    numerator_df = symbol_data.get(numerator_symbol)
    denominator_df = symbol_data.get(denominator_symbol)
    
    # USDC special handling - if USDC is missing, create a synthetic dataframe with value 1.0
    if denominator_symbol == "USDC" and (denominator_df is None or denominator_df.empty):
        print(f"Creating synthetic USDC data as fallback (value = 1.0)")
        
        # If we have numerator data, use its index for USDC
        if numerator_df is not None and not numerator_df.empty:
            # Create a DataFrame with same index as numerator but all values = 1.0
            denominator_df = pd.DataFrame(index=numerator_df.index, data={'close': 1.0})
            # Add to symbol_data so it can be reused
            symbol_data[denominator_symbol] = denominator_df
        else:
            print(f"Cannot create synthetic USDC data because {numerator_symbol} data is missing")
            return None
    
    if numerator_df is None or denominator_df is None:
        print(f"Missing data for {numerator_symbol} or {denominator_symbol}")
        if numerator_df is None:
            print(f"No data for {numerator_symbol}")
        if denominator_df is None:
            print(f"No data for {denominator_symbol}")
        return None
    
    print(f"Merging data for {numerator_symbol}/{denominator_symbol}")
    print(f"{numerator_symbol} date range: {numerator_df.index.min()} to {numerator_df.index.max()}, {len(numerator_df)} points")
    print(f"{denominator_symbol} date range: {denominator_df.index.min()} to {denominator_df.index.max()}, {len(denominator_df)} points")
    
    # Merge dataframes on datetime index
    merged_df = pd.merge(
        numerator_df, 
        denominator_df, 
        left_index=True, 
        right_index=True,
        suffixes=(f'_{numerator_symbol}', f'_{denominator_symbol}')
    )
    
    if merged_df.empty:
        print(f"No overlapping data points for {numerator_symbol}/{denominator_symbol}")
        return None
    
    print(f"After merge: {len(merged_df)} overlapping data points")
    
    # Calculate pair price
    merged_df['price'] = merged_df[f'close_{numerator_symbol}'] / merged_df[f'close_{denominator_symbol}']
    
    # Remove any rows with infinite or NaN values
    merged_df = merged_df[np.isfinite(merged_df['price'])]
    
    if merged_df.empty:
        print(f"No valid price data for {numerator_symbol}/{denominator_symbol} after filtering")
        return None
    
    # Ensure data is sorted by datetime
    merged_df = merged_df.sort_index()
    
    # Only keep the price column - IMPORTANT: Not resampling to preserve hourly data
    result_df = pd.DataFrame(merged_df['price'])
    
    # Drop any remaining NaN values
    result_df = result_df.dropna()
    
    print(f"Final data for {numerator_symbol}/{denominator_symbol}: {len(result_df)} points")
    
    return result_df

def fetch_price_data(trading_pair, symbol_data, pair_configs):
    """
    Get price data for a trading pair from the symbol data.
    
    Args:
        trading_pair (str): Trading pair name (e.g., "SOL/USDC")
        symbol_data (dict): Dictionary mapping symbols to their price dataframes
        pair_configs (list): List of pair configuration dictionaries
        
    Returns:
        pandas.DataFrame: Dataframe with timestamp and price columns
    """
    # Find the configuration for this pair
    pair_config = next((config for config in pair_configs if config["name"] == trading_pair), None)
    
    if pair_config is None:
        raise ValueError(f"No configuration found for {trading_pair}")
    
    # Calculate the pair price
    return calculate_pair_price(
        symbol_data, 
        pair_config["numerator"], 
        pair_config["denominator"]
    )

def calculate_returns(prices):
    """
    Calculate log returns from price data, preserving hourly granularity.
    
    Args:
        prices (pandas.Series): Price series
        
    Returns:
        pandas.Series: Log returns series
    """
    # Create a copy to avoid affecting the original Series
    prices_copy = prices.copy()
    
    # Remove any zero values which would cause problems with log returns
    prices_copy = prices_copy[prices_copy > 0]
    
    if len(prices_copy) < 2:
        print("Not enough non-zero prices to calculate returns")
        return pd.Series()
    
    # IMPORTANT: We do NOT resample to daily frequency to preserve hourly data
    
    # Calculate log returns
    try:
        returns = np.log(prices_copy / prices_copy.shift(1)).dropna()
        
        # Cap extreme values that might be due to data errors (5 standard deviations)
        if len(returns) > 0:
            std = returns.std()
            if not np.isnan(std) and std > 0:
                threshold = 5 * std
                original_len = len(returns)
                returns = returns.clip(-threshold, threshold)
                print(f"Clipped returns outside of ±{threshold:.6f} ({original_len - len(returns)} points affected)")
        
        return returns
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        return pd.Series()

def calculate_volatility(returns, window_hours, annualize=True):
    """
    Calculate rolling volatility from returns using hourly data.
    
    Args:
        returns (pandas.Series): Hourly returns series
        window_hours (int): Lookback window in hours
        annualize (bool): Whether to annualize the volatility
        
    Returns:
        pandas.Series: Rolling volatility series
    """
    # Ensure we have a valid window size
    if window_hours < 2:
        print(f"WARNING: Window size {window_hours} is too small, setting to 2")
        window_hours = 2
    
    # Calculate rolling standard deviation using the specified hour window
    min_periods = max(2, window_hours // 4)  # Require at least 25% of window size or 2 observations
    vol = returns.rolling(window=window_hours, min_periods=min_periods).std()
    
    # Annualize from hourly to annual volatility by multiplying by sqrt(hours in a year)
    if annualize:
        vol = vol * np.sqrt(TRADING_HOURS_PER_YEAR)
    
    return vol

def calculate_ewma_volatility(returns, lambda_=0.94, annualize=True):
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    This is a good proxy for instantaneous volatility as it gives more weight to recent observations.
    
    Args:
        returns (pandas.Series): Hourly returns series
        lambda_ (float): Decay factor (RiskMetrics typically uses 0.94 for daily data)
        annualize (bool): Whether to annualize the volatility
        
    Returns:
        pandas.Series: EWMA volatility series
    """
    # Square the returns
    returns_squared = returns ** 2
    
    # Calculate EWMA of squared returns
    ewma = returns_squared.ewm(alpha=(1-lambda_)).mean()
    
    # Take the square root to get volatility
    vol = np.sqrt(ewma)
    
    # Annualize from hourly to annual volatility by multiplying by sqrt(hours in a year)
    if annualize:
        vol = vol * np.sqrt(TRADING_HOURS_PER_YEAR)
    
    return vol

def calculate_future_volatility(returns, window_hours, annualize=True):
    """
    Calculate future volatility for the next window_hours using hourly data.
    Uses the exact same methodology as lookback volatility for consistency.
    
    Args:
        returns (pandas.Series): Hourly returns series
        window_hours (int): Future window size in hours
        annualize (bool): Whether to annualize the volatility
        
    Returns:
        pandas.Series: Future volatility series
    """
    # Ensure we have a valid window size
    if window_hours < 2:
        print(f"WARNING: Window size {window_hours} is too small, setting to 2")
        window_hours = 2
    
    # Gather future returns for the specified window
    future_returns = []
    for i in range(1, window_hours + 1):
        shifted = returns.shift(-i)
        future_returns.append(shifted)
    
    # Stack all future returns
    future_return_matrix = pd.concat(future_returns, axis=1)
    
    # Calculate standard deviation across the future window
    future_vol = future_return_matrix.std(axis=1, skipna=False)
    
    # Annualize from hourly to annual volatility by multiplying by sqrt(hours in a year)
    if annualize:
        vol = future_vol * np.sqrt(TRADING_HOURS_PER_YEAR)
        return vol
    
    return future_vol

def verify_volatility_consistency(returns, results):
    """
    Verify that future volatility at time t matches past volatility at time t+24h.
    This helps validate that our calculations are consistent.
    
    Args:
        returns (pandas.Series): Returns series
        results (pandas.DataFrame): Dataframe with volatility results
        
    Returns:
        dict: Validation statistics
    """
    if 'future_vol_1_day' not in results.columns or 'vol_1_day' not in results.columns:
        print("Missing required columns for validation")
        return None
    
    # Get all datetimes in ascending order
    datetimes = sorted(results.index)
    
    # We need at least 24 hours of data to perform validation
    if len(datetimes) <= 24:
        print("Not enough data for validation")
        return None
    
    validation_data = []
    
    for i, dt in enumerate(datetimes[:-24]):  # Skip last 24 hours
        # Look for hourly timestamp that's 24 hours ahead
        target_dt = dt + pd.Timedelta(hours=24)
        
        # Find if we have a data point at exactly 24 hours ahead
        match_idx = None
        for j in range(i+1, len(datetimes)):
            if datetimes[j] == target_dt:
                match_idx = j
                break
                
        # If we found an exact match
        if match_idx is not None:
            future_vol = results.loc[dt, 'future_vol_1_day']
            future_lookback_vol = results.loc[datetimes[match_idx], 'vol_1_day']
            
            validation_data.append({
                'datetime': dt,
                'future_datetime': datetimes[match_idx],
                'future_vol': future_vol,
                'future_lookback_vol': future_lookback_vol,
                'difference': future_vol - future_lookback_vol,
                'rel_error_pct': ((future_vol - future_lookback_vol) / future_lookback_vol * 100) 
                                if future_lookback_vol != 0 else float('nan')
            })
    
    if not validation_data:
        print("No matching timestamps found for validation")
        return None
        
    # Calculate validation metrics
    validation_df = pd.DataFrame(validation_data)
    mae = validation_df['difference'].abs().mean()
    mape = validation_df['rel_error_pct'].abs().mean()
    
    return {
        'validation_df': validation_df,
        'mae': mae,
        'mape': mape
    }

def analyze_trading_pair(trading_pair, symbol_data, pair_configs, lambda_value=0.94):
    """
    Analyze volatility for a single trading pair using hourly data.
    
    Args:
        trading_pair (str): Trading pair name
        symbol_data (dict): Dictionary mapping symbols to their price dataframes
        pair_configs (list): List of pair configuration dictionaries
        lambda_value (float): Lambda value for EWMA calculation (0.94 is standard)
        
    Returns:
        tuple: (results DataFrame, price_data DataFrame) or (None, None) if analysis fails
    """
    print(f"Analyzing {trading_pair}...")
    
    try:
        # Fetch price data for the trading pair
        price_data = fetch_price_data(trading_pair, symbol_data, pair_configs)
        
        if price_data is None or price_data.empty:
            print(f"No valid price data for {trading_pair}")
            return None, None
        
        # Print some basic stats about the price data
        print(f"Price data for {trading_pair}: {len(price_data)} points from {price_data.index.min().date()} to {price_data.index.max().date()}")
        print(f"Price range: {price_data['price'].min()} to {price_data['price'].max()}, mean: {price_data['price'].mean()}")
        
        # Calculate returns - preserving hourly granularity
        returns = calculate_returns(price_data['price'])
        
        if returns.empty:
            print(f"No valid returns for {trading_pair}")
            return None, None
        
        # Print data frequency for verification
        date_strings = [d.strftime('%Y-%m-%d') for d in returns.index]
        unique_days = len(set(date_strings))
        freq_ratio = len(returns) / unique_days if unique_days > 0 else 0
        print(f"Data frequency check: {len(returns)} points over {unique_days} days = {freq_ratio:.2f} points/day")
        print(f"Calculated {len(returns)} returns for {trading_pair}")
        print(f"Return stats - min: {returns.min()}, max: {returns.max()}, mean: {returns.mean()}, std: {returns.std()}")
        
        # Initialize results dataframe
        results = pd.DataFrame(index=returns.index)
        
        # Calculate volatility for each lookback period using hourly windows
        for period_name, hours in lookback_periods.items():
            # Only calculate if we have enough data
            if len(returns) >= hours:
                vol = calculate_volatility(returns, hours)
                results[f"vol_{period_name.replace(' ', '_')}"] = vol
                print(f"Calculated {period_name} volatility with {len(vol)} points using {hours} hours window")
            else:
                print(f"Not enough data for {period_name} lookback for {trading_pair} (need {hours}, have {len(returns)})")
        
        # Calculate EWMA volatility (as a proxy for instantaneous volatility) with the specified lambda
        results['vol_ewma'] = calculate_ewma_volatility(returns, lambda_=lambda_value)
        print(f"Calculated EWMA volatility with {len(results['vol_ewma'])} points using lambda={lambda_value}")
        
        # Calculate 1-day future volatility (24 hours) - using EXACT same methodology as lookback
        results['future_vol_1_day'] = calculate_future_volatility(returns, 24)  # 24 hours
        print(f"Calculated future volatility with {len(results['future_vol_1_day'])} points using 24 hours window")
        
        # Verify that future volatility at time t matches past volatility at time t+24h
        validation = verify_volatility_consistency(returns, results)
        if validation is not None:
            # Add validation statistics to the console output
            print(f"Volatility consistency check - Mean Absolute Error: {validation['mae']:.6f}")
            print(f"Volatility consistency check - Mean Relative Error: {validation['mape']:.2f}%")
        
        # Handle NaN values - keep rows with at least one valid lookback volatility
        original_len = len(results)
        lookback_cols = [col for col in results.columns if col != 'future_vol_1_day']
        valid_indices = results[lookback_cols].dropna(how='all').index
        results = results.loc[valid_indices]
        print(f"Dropped {original_len - len(results)} rows with all NaN lookback volatilities, {len(results)} rows remaining")
        
        if results.empty:
            print(f"No valid volatility results for {trading_pair} after filtering NaN values")
            return None, None
        
        # Final check - ensure we have usable data for analysis
        if len(results) < 10:  # Arbitrary threshold
            print(f"Insufficient data points ({len(results)}) for reliable volatility analysis of {trading_pair}")
            return None, None
            
        print(f"Final volatility results for {trading_pair}: {len(results)} data points")
        return results, price_data
    
    except Exception as e:
        print(f"Error analyzing {trading_pair}: {str(e)}")
        traceback.print_exc()
        return None, None
    
def plot_r_squared_bar_chart(all_r_squared):
    """
    Create an interactive Plotly bar chart for R² values.
    
    Args:
        all_r_squared (dict): Dictionary with R² results for each trading pair
        
    Returns:
        plotly.graph_objects.Figure: Interactive bar chart
    """
    # Create a DataFrame for R² values
    r2_df = pd.DataFrame(all_r_squared)
    
    # Calculate mean R² values, skipping NaN values
    avg_r2 = r2_df.mean(axis=1, skipna=True)
    
    # Sort by R² value
    avg_r2 = avg_r2.sort_values()
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=avg_r2.index,
        x=avg_r2.values,
        orientation='h',
        marker=dict(
            color=avg_r2.values,
            colorscale='Viridis',
            cmin=0,
            cmax=1
        ),
        text=[f"{v:.3f}" for v in avg_r2.values],
        textposition='outside',
        hovertemplate='%{y}<br>R²: %{x:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Average R² with Future Volatility',
        xaxis_title='R² Value',
        yaxis_title='Lookback Period',
        margin=dict(l=40, r=100, t=60, b=40),
        xaxis=dict(range=[0, max(max(avg_r2.values) + 0.1, 1)]),
    )
    
    return fig


def calculate_correlations(results):
    """
    Calculate correlations between lookback volatilities and future volatility.
    
    Args:
        results (pandas.DataFrame): Dataframe with volatility results
        
    Returns:
        pandas.Series: Correlations with future volatility
    """
    correlations = {}
    
    # Check if future volatility column exists
    if 'future_vol_1_day' not in results.columns:
        print("Warning: future_vol_1_day column not found in results")
        return pd.Series(correlations)
    
    # Get all columns except future volatility
    lookback_cols = [col for col in results.columns if col != 'future_vol_1_day']
    
    # For each lookback column, calculate correlation with future volatility
    for col in lookback_cols:
        # Select only rows where both values are not NaN
        mask = results[col].notna() & results['future_vol_1_day'].notna()
        if mask.sum() > 5:  # Need at least 5 points for meaningful correlation
            corr = results.loc[mask, col].corr(results.loc[mask, 'future_vol_1_day'])
            correlations[col.replace('vol_', '').replace('_', ' ')] = corr
        else:
            print(f"Not enough valid data points for correlation between {col} and future_vol_1_day")
            correlations[col.replace('vol_', '').replace('_', ' ')] = float('nan')
    
    return pd.Series(correlations)

def plot_volatility_comparison_plotly(results, trading_pair):
    """
    Create an interactive Plotly chart comparing different volatility measures.
    
    Args:
        results (pandas.DataFrame): Dataframe with volatility results
        trading_pair (str): Trading pair name
        
    Returns:
        plotly.graph_objects.Figure: Interactive volatility comparison chart
    """
    fig = go.Figure()
    
    # Determine color palette for consistent colors
    colorscale = px.colors.qualitative.Bold
    
    # Get columns in a priority order to ensure consistent coloring
    all_cols = []
    priority_cols = ['vol_ewma', 'vol_1_day', 'vol_3_days', 'vol_1_week', 'vol_2_weeks']
    
    # First add priority columns that exist
    for col in priority_cols:
        if col in results.columns and results[col].notna().any():
            all_cols.append(col)
    
    # Then add any other columns not in priority list
    for col in results.columns:
        if col != 'future_vol_1_day' and col not in all_cols and results[col].notna().any():
            all_cols.append(col)
    
    # Add traces for each volatility measure with better naming and styling
    for i, col in enumerate(all_cols):
        display_name = col.replace('vol_', '').replace('_', ' ')
        if display_name == 'ewma':
            display_name = 'EWMA (Instantaneous)'
            
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results[col],
            mode='lines',
            name=display_name,
            line=dict(
                color=colorscale[i % len(colorscale)],
                width=2
            ),
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
        ))
    
    # Add future volatility as a dashed line if it exists
    if 'future_vol_1_day' in results.columns and results['future_vol_1_day'].notna().any():
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['future_vol_1_day'],
            mode='lines',
            name='Future 1-day volatility',
            line=dict(color='black', dash='dash', width=1.5),
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
        ))
    
    # Update layout for better appearance
    fig.update_layout(
        title=f'Volatility Comparison for {trading_pair}',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white',
        font=dict(family="Arial, sans-serif")
    )
    
    # Add range selector for time periods
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    # Add cleaner grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    
    return fig

def plot_correlation_heatmap_plotly(all_correlations):
    """
    Create an interactive Plotly heatmap of correlations.
    
    Args:
        all_correlations (dict): Dictionary with correlation results for each trading pair
        
    Returns:
        plotly.graph_objects.Figure: Interactive correlation heatmap
    """
    # Create a dataframe with correlations for all trading pairs
    corr_df = pd.DataFrame(all_correlations).T
    
    # Create the heatmap
    fig = px.imshow(
        corr_df,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(x="Trading Pair", y="Lookback Period", color="Correlation")
    )
    
    # Customize the layout
    fig.update_layout(
        title='Correlation of Lookback Volatilities with Future 1-day Volatility',
        coloraxis_colorbar=dict(
            title="Correlation",
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Add correlation values as text
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            value = corr_df.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 0.5 else 'black'
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color=text_color)
                )
    
    return fig

def plot_average_correlations_plotly(all_correlations):
    """
    Create an interactive Plotly bar chart for average correlations.
    
    Args:
        all_correlations (dict): Dictionary with correlation results for each trading pair
        
    Returns:
        plotly.graph_objects.Figure: Interactive bar chart
    """
    # Create a DataFrame for correlations
    corr_df = pd.DataFrame(all_correlations)
    
    # Calculate mean correlations, skipping NaN values
    avg_correlations = corr_df.mean(axis=1, skipna=True)
    
    # Sort by correlation value
    avg_correlations = avg_correlations.sort_values()
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=avg_correlations.index,
        x=avg_correlations.values,
        orientation='h',
        marker=dict(
            color=avg_correlations.values,
            colorscale='RdBu_r',
            cmin=-1,
            cmax=1
        ),
        text=[f"{v:.2f}" for v in avg_correlations.values],
        textposition='outside',
        hovertemplate='%{y}<br>Correlation: %{x:.4f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Average Correlation with Future Volatility',
        xaxis_title='Correlation',
        yaxis_title='Lookback Period',
        margin=dict(l=40, r=100, t=60, b=40),
        xaxis=dict(range=[min(min(avg_correlations.values) - 0.1, -1), max(max(avg_correlations.values) + 0.1, 1)]),
    )
    
    return fig

def calculate_r_squared(results):
    """
    Calculate R² (coefficient of determination) between lookback volatilities and future volatility.
    
    Args:
        results (pandas.DataFrame): Dataframe with volatility results
        
    Returns:
        pandas.Series: R² values with future volatility
    """
    r_squared = {}
    
    # Check if future volatility column exists
    if 'future_vol_1_day' not in results.columns:
        print("Warning: future_vol_1_day column not found in results")
        return pd.Series(r_squared)
    
    # Get all columns except future volatility
    lookback_cols = [col for col in results.columns if col != 'future_vol_1_day']
    
    # For each lookback column, calculate R² with future volatility
    for col in lookback_cols:
        # Select only rows where both values are not NaN
        mask = results[col].notna() & results['future_vol_1_day'].notna()
        
        if mask.sum() > 5:  # Need at least 5 points for meaningful calculation
            # Get the data
            X = results.loc[mask, col]
            y = results.loc[mask, 'future_vol_1_day']
            
            # Calculate R² (square of correlation coefficient)
            correlation = X.corr(y)
            r_squared[col.replace('vol_', '').replace('_', ' ')] = correlation ** 2
        else:
            print(f"Not enough valid data points for R² between {col} and future_vol_1_day")
            r_squared[col.replace('vol_', '').replace('_', ' ')] = float('nan')
    
    return pd.Series(r_squared)

def suggest_optimal_range(vol, price, confidence=0.95):
    """
    Suggest an optimal price range for Uniswap V3 based on volatility.
    
    Args:
        vol (float): Annualized volatility
        price (float): Current price
        confidence (float): Confidence level for the range
        
    Returns:
        tuple: (lower_bound, upper_bound) for the suggested price range
    """
    # Convert annualized volatility to daily
    # Since our volatility is annualized from hourly data, we divide by sqrt(hours in a year)
    # and multiply by sqrt(24) to get daily volatility
    daily_vol = vol / np.sqrt(TRADING_HOURS_PER_YEAR) * np.sqrt(24)
    
    # Calculate z-score for the confidence level
    z_score = norm.ppf((1 + confidence) / 2)
    
    # Calculate price range
    lower_bound = price * np.exp(-z_score * daily_vol)
    upper_bound = price * np.exp(z_score * daily_vol)
    
    return lower_bound, upper_bound

def analyze_optimal_ranges(results, price_data, trading_pair, confidence=0.95):
    """
    Analyze optimal price ranges for Uniswap V3 based on different volatility estimates.
    
    Args:
        results (pandas.DataFrame): Dataframe with volatility results
        price_data (pandas.DataFrame): Dataframe with price data
        trading_pair (str): Trading pair name
        confidence (float): Confidence level for the range (0.95 is standard)
        
    Returns:
        dict: Dictionary with optimal ranges for each volatility measure
    """
    try:
        # Get the most recent data point
        latest_date = results.index[-1]
        latest_price = price_data.loc[price_data.index[-1], 'price']
        
        ranges = {}
        
        print(f"\nOptimal price ranges for {trading_pair} (current price: {latest_price:.6f}, confidence: {confidence:.2f}):")
        
        for col in results.columns:
            if col != 'future_vol_1_day':
                vol = results.loc[latest_date, col]
                lower, upper = suggest_optimal_range(vol, latest_price, confidence)
                range_width = (upper - lower) / latest_price * 100  # as percentage of current price
                
                ranges[col.replace('vol_', '').replace('_', ' ')] = {
                    'volatility': vol,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'range_width_pct': range_width
                }
                
                print(f"{col.replace('vol_', '').replace('_', ' ')}:")
                print(f"  Volatility: {vol:.6f}")
                print(f"  Suggested range: {lower:.6f} to {upper:.6f} ({range_width:.2f}% width)")
        
        return ranges
    
    except Exception as e:
        print(f"Error calculating optimal ranges for {trading_pair}: {str(e)}")
        return {}
    
def create_volatility_validation_chart(results):
    """
    Create a chart that demonstrates the relationship between future and past volatility.
    
    Args:
        results (pandas.DataFrame): DataFrame with volatility results
        
    Returns:
        plotly.graph_objects.Figure: Chart comparing future and shifted past volatility
    """
    if 'future_vol_1_day' not in results.columns or 'vol_1_day' not in results.columns:
        print("Missing required columns for validation chart")
        return None
        
    # Create a copy of results for manipulation
    chart_data = results.copy()
    
    # Shift past volatility back by 24 hours (approximate - depends on data frequency)
    # For hourly data, we need to shift by 24 periods
    chart_data['vol_1_day_shifted_back'] = chart_data['vol_1_day'].shift(-24)
    
    # Create the chart
    fig = go.Figure()
    
    # Add future volatility at time t
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['future_vol_1_day'],
        mode='lines',
        name='Future 1-day Volatility at time t',
        line=dict(color='blue', width=2)
    ))
    
    # Add past volatility at time t+24h
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['vol_1_day_shifted_back'],
        mode='lines',
        name='Past 1-day Volatility at time t+24h',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Update layout
    fig.update_layout(
        title='Validation: Future Volatility vs. Shifted Past Volatility',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    
    return fig

def create_volatility_selector_dashboard(all_results, selected_pair):
    """
    Create a customizable dashboard where users can select which volatility measures to display.
    
    Args:
        all_results (dict): Dictionary with results for each trading pair
        selected_pair (str): Currently selected trading pair
        
    Returns:
        None: This function uses Streamlit directly
    """
    if selected_pair not in all_results or all_results[selected_pair] is None:
        st.warning(f"No data available for {selected_pair}")
        return
    
    results = all_results[selected_pair]
    
    # Get available volatility measures (excluding future volatility)
    available_measures = [col for col in results.columns if col != 'future_vol_1_day']
    
    # Create better display names for the measures
    display_names = {col: col.replace('vol_', '').replace('_', ' ').title() for col in available_measures}
    display_names['vol_ewma'] = 'EWMA (Instantaneous)'
    
    # Use the display names for the multi-select
    selected_display_names = st.multiselect(
        "Select Volatility Measures to Display:",
        options=list(display_names.values()),
        default=['EWMA (Instantaneous)', '1 Day', '1 Week'] if all(m in display_names.values() for m in ['EWMA (Instantaneous)', '1 Day', '1 Week']) else list(display_names.values())[:min(3, len(display_names))],
    )
    
    # Map selected display names back to column names
    reverse_mapping = {v: k for k, v in display_names.items()}
    selected_measures = [reverse_mapping[name] for name in selected_display_names]
    
    # Add toggle for future volatility
    show_future = st.checkbox("Show Future 1-day Volatility", value=True)
    
    # Create a filtered dataframe with only the selected measures
    filtered_df = results[selected_measures].copy()
    
    # Add future volatility if selected
    if show_future and 'future_vol_1_day' in results.columns:
        filtered_df['future_vol_1_day'] = results['future_vol_1_day']
    
    # Create custom Plotly chart with the selected measures
    fig = go.Figure()
    
    # Use a consistent color palette
    colors = px.colors.qualitative.Bold
    
    # Add traces for selected volatility measures
    for i, col in enumerate(selected_measures):
        display_name = display_names[col]
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df[col],
            mode='lines',
            name=display_name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
        ))
    
    # Add future volatility if selected
    if show_future and 'future_vol_1_day' in filtered_df.columns:
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['future_vol_1_day'],
            mode='lines',
            name='Future 1-day Volatility',
            line=dict(color='black', dash='dash', width=1.5),
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Custom Volatility View for {selected_pair}',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        template='plotly_white',
        height=500,
    )
    
    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ]),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    
    # Cleaner grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add volatility validation chart to compare future vs. actual
    if 'vol_1_day' in results.columns and 'future_vol_1_day' in results.columns:
        with st.expander("Volatility Forecast Validation"):
            st.markdown("""
            ### Volatility Forecast Validation
            
            This chart compares the 1-day future volatility forecast (black dashed line) with the actual 1-day realized volatility (solid line) to validate the forecasting accuracy.
            
            In theory, the future 1-day volatility on day X should match the 1-day lookback volatility on day X+1. Significant differences may indicate market regime changes or calculation inconsistencies.
            """)
            
            # Create validation chart comparing future_vol_1_day with the next day's vol_1_day
            val_fig = go.Figure()
            
            # Add future volatility
            val_fig.add_trace(go.Scatter(
                x=results.index,
                y=results['future_vol_1_day'],
                mode='lines',
                name='Future 1-day Volatility (Forecast)',
                line=dict(color='black', dash='dash', width=1.5),
                hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
            ))
            
            # Add 1-day lookback volatility (which should match future_vol of previous day)
            val_fig.add_trace(go.Scatter(
                x=results.index,
                y=results['vol_1_day'],
                mode='lines',
                name='1-day Volatility (Actual)',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
            ))
            
            # Update layout
            val_fig.update_layout(
                title='Volatility Forecast vs. Actual',
                xaxis_title='Date',
                yaxis_title='Annualized Volatility',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template='plotly_white'
            )
            
            st.plotly_chart(val_fig, use_container_width=True)
            
            # Add validation view with shifted data to better compare
            validation_fig = create_volatility_validation_chart(results)
            if validation_fig:
                st.plotly_chart(validation_fig, use_container_width=True)
                st.info("This chart shows the future volatility prediction at time t compared to the past volatility at time t+24h. If our calculations are consistent, these lines should match closely.")
            
            # Calculate forecast error metrics
            if len(results) > 1:
                # Calculate forecast error (future vol - next day's actual vol)
                future_vol = results['future_vol_1_day'][:-24]  # Exclude last 24 hours
                next_day_vol = results['vol_1_day'][24:]  # Start from 24 hours ahead
                
                # Make sure indexes match
                if len(future_vol) == len(next_day_vol):
                    next_day_vol.index = future_vol.index
                    forecast_error = future_vol - next_day_vol
                    
                    # Calculate error metrics
                    mae = np.abs(forecast_error).mean()
                    mape = np.abs(forecast_error / next_day_vol).mean() * 100
                    
                    # Display metrics
                    st.markdown(f"""
                    #### Forecast Error Metrics
                    
                    - **Mean Absolute Error (MAE):** {mae:.4f}
                    - **Mean Absolute Percentage Error (MAPE):** {mape:.2f}%
                    """)
                    
                    # Create histogram of forecast errors
                    error_fig = px.histogram(
                        forecast_error,
                        nbins=20,
                        labels={'value': 'Forecast Error'},
                        title='Distribution of Volatility Forecast Errors'
                    )
                    
                    st.plotly_chart(error_fig, use_container_width=True)

def expanded_volatility_explanation():
    """
    Returns a detailed markdown explanation of volatility calculation methods,
    with special focus on how EWMA serves as an indicator of instantaneous volatility.
    """
    return """
    ## Volatility Calculation Methodology
    
    The volatility shown in the charts is calculated using several different approaches:

    ### 1. Historical Window Volatility

    The traditional approach uses a fixed historical window (lookback period) to calculate volatility:
    
    * **Returns Calculation**: Log returns are calculated from price data: $ r_t = \ln(P_t / P_{t-1}) $
    * **Rolling Window**: Standard deviation is computed over rolling windows of different lengths (1 day, 3 days, 1 week, 2 weeks)
    * **Annualization**: The standard deviation is annualized by multiplying by $ \sqrt{trading\ days\ per\ year} $

    ### 2. EWMA Volatility as Instantaneous Volatility Indicator

    **Exponentially Weighted Moving Average (EWMA)** provides a better proxy for instantaneous volatility compared to traditional fixed window approaches. Here's why:

    #### How EWMA Works

    EWMA calculates volatility by weighting returns based on recency, with the formula:

    $$ \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_t^2 $$

    Where:
    * $ \sigma_t^2 $ is the variance at time t
    * $ \lambda $ is the decay factor (typically 0.94 for daily data)
    * $ r_t $ is the return at time t

    #### Why EWMA Is Superior for Instantaneous Volatility

    1. **Recency Bias**: EWMA gives more weight to recent observations and less weight to older ones, making it more responsive to current market conditions
    
    2. **Automatic Mean Reversion**: EWMA naturally adjusts faster during periods of changing volatility, reacting more quickly to volatility spikes
    
    3. **Memory Decay**: The influence of past returns decays exponentially, mimicking how markets "forget" older price movements
    
    4. **Smoother Transitions**: Unlike simple moving windows that can have sharp transitions when large returns enter or exit the window, EWMA provides smoother estimates
    
    5. **Industry Standard**: RiskMetrics and many financial institutions use EWMA (with λ=0.94 for daily data) as their standard approach for estimating current volatility

    #### Mathematical Intuition

    At λ=0.94, the weight on a return from:
    * Today: 6% weight
    * 1 week ago: ~3% weight
    * 1 month ago: ~0.7% weight
    
    This creates a naturally declining influence curve that can better capture the current market volatility regime.

    ### 3. Future Volatility

    To evaluate how well each method predicts volatility, we also calculate:
    
    * **Future Volatility**: The actual volatility observed in the following day(s), used as a benchmark to compare the predictive power of different lookback periods
    
    The ideal method is the one that correlates best with future volatility.
    """

def run_streamlit_dashboard():
    """
    Run the Streamlit dashboard for crypto volatility analysis.
    """
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Please install it with 'pip install streamlit'.")
        return
    
    # Add a title that explains the database connection requirements
    st.set_page_config(
        page_title="Crypto Volatility Analysis for Uniswap V3",
        page_icon="📊",
        layout="wide",
    )
    
    # Initialize database connection
    global db_connection
    if db_connection is None:
        db_connection = setup_timescale_connection()
    
    # Page header
    st.title("Crypto Volatility Analysis for Uniswap V3")
    st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 0.3rem;
    }
    </style>
    <div class="subheader">
    Analyze volatility patterns to find optimal liquidity provision ranges
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for the top section
    status_col, config_col = st.columns([1, 2])
    
    with status_col:
        # Add connection status with nicer styling
        if db_connection is None:
            st.error("⚠️ Database: Not Connected")
        else:
            st.success("✅ Database: Connected")
    
    with config_col:
        # Select months of data to analyze
        months_to_analyze = st.slider("Historical Data Range (Months)", 1, 6, 3)
    
    # Sidebar for configuration options
    with st.sidebar:
        st.header("Trading Pairs")
        
        # Group trading pairs by base currency
        eth_pairs = [pair for pair in trading_pairs if "ETH" in pair]
        sol_pairs = [pair for pair in trading_pairs if "SOL" in pair]
        other_pairs = [pair for pair in trading_pairs if not any(x in pair for x in ["ETH", "SOL"])]
        
        # Create expanders for each group
        selected_pairs = []
        
        with st.expander("ETH Pairs", expanded=True):
            eth_selected = {}
            for pair in eth_pairs:
                eth_selected[pair] = st.checkbox(
                    pair, 
                    value=(pair in ["ETH/USDC", "ETH/BTC"][:min(2, len(eth_pairs))]),
                    key=f"eth_{pair.replace('/', '_')}"
                )
            selected_pairs.extend([pair for pair, selected in eth_selected.items() if selected])
        
        with st.expander("SOL Pairs", expanded=True):
            sol_selected = {}
            for pair in sol_pairs:
                sol_selected[pair] = st.checkbox(
                    pair, 
                    value=(pair == "SOL/USDC" and "SOL/USDC" in sol_pairs),
                    key=f"sol_{pair.replace('/', '_')}"
                )
            selected_pairs.extend([pair for pair, selected in sol_selected.items() if selected])
        
        if other_pairs:
            with st.expander("Other Pairs"):
                other_selected = {}
                for pair in other_pairs:
                    other_selected[pair] = st.checkbox(
                        pair, 
                        value=False,
                        key=f"other_{pair.replace('/', '_')}"
                    )
                selected_pairs.extend([pair for pair, selected in other_selected.items() if selected])
        
        # Add a divider
        st.markdown("---")
        
        # Volatility calculation settings
        st.header("Analysis Settings")
        
        # Lambda value for EWMA
        lambda_value = st.slider(
            "EWMA Lambda Value", 
            min_value=0.8, 
            max_value=0.99, 
            value=0.94, 
            step=0.01,
            help="Higher lambda gives more weight to historical data (0.94 is RiskMetrics standard). Lower values make EWMA more responsive to recent price changes."
        )
        
        # Confidence level for price ranges
        confidence_level = st.slider(
            "Range Confidence Level", 
            min_value=0.5, 
            max_value=0.99, 
            value=0.95, 
            step=0.01,
            help="Higher confidence level results in wider price ranges (95% is standard)"
        )
        
        st.markdown("---")
        st.header("Comparison View")
        
        # Create multi-select for comparison
        comparison_pairs = st.multiselect(
            "Select pairs to compare:", 
            options=trading_pairs,
            default=["SOL/USDC", "ETH/USDC"][:min(2, len(trading_pairs))]
        )
    
        # Add a run button
        st.markdown("---")
        run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Placeholder for trading pair selection if none selected
    if not selected_pairs:
        st.warning("Please select at least one trading pair from the sidebar to begin analysis.")
        selected_pairs = []        
    
    # Initialize session state
    if 'symbol_data' not in st.session_state:
        st.session_state.symbol_data = None
    if 'all_results' not in st.session_state:
        st.session_state.all_results = {}
    if 'all_price_data' not in st.session_state:
        st.session_state.all_price_data = {}
    if 'all_correlations' not in st.session_state:
        st.session_state.all_correlations = {}
    if 'all_ranges' not in st.session_state:
        st.session_state.all_ranges = {}
    
    # Function to run the analysis
    def run_volatility_analysis():
        """
        Run the volatility analysis for selected trading pairs.
        """
        # Check if we have a valid database connection
        if db_connection is None:
            st.error("No connection to database. Please check your credentials.")
            return
            
        # Create a collapsible container for progress
        with st.expander("Analysis Progress", expanded=True):
            # First step: fetch data
            status_container = st.empty()
            progress_bar = st.progress(0)
            symbol_status = st.empty()
            
            status_container.info("Step 1/3: Fetching cryptocurrency data...")
            
            # Determine which symbols we need based on selected pairs
            needed_symbols = set()
            for pair in selected_pairs:
                # Find configuration for this pair
                pair_config = next((config for config in trading_pairs_config if config["name"] == pair), None)
                if pair_config:
                    needed_symbols.add(pair_config["numerator"])
                    needed_symbols.add(pair_config["denominator"])
            
            # Fetch only the symbols we need
            needed_symbols = list(needed_symbols)
            st.session_state.symbol_data = {}
            
            # Fetch data for each symbol with progress updates
            for i, symbol in enumerate(needed_symbols):
                symbol_progress = i / len(needed_symbols)
                progress_bar.progress(symbol_progress * 0.4)  # 40% of progress for data fetching
                symbol_status.info(f"Fetching data for {symbol}...")
                
                try:
                    symbol_data = fetch_symbol_data(symbol, months=months_to_analyze, connection=db_connection)
                    st.session_state.symbol_data[symbol] = symbol_data
                except Exception as e:
                    st.error(f"Error fetching {symbol}: {str(e)}")
            
            # Step 2: Analysis
            status_container.info("Step 2/3: Analyzing volatility patterns...")
            progress_bar.progress(0.4)  # 40% complete after data fetching
            
            # Reset results
            st.session_state.all_results = {}
            st.session_state.all_price_data = {}
            st.session_state.all_correlations = {}
            st.session_state.all_r_squared = {} 
            st.session_state.all_ranges = {}
            
            # Analyze each selected trading pair
            for i, trading_pair in enumerate(selected_pairs):
                pair_progress = 0.4 + (i / len(selected_pairs) * 0.5)  # 50% of progress for analysis
                progress_bar.progress(pair_progress)
                symbol_status.info(f"Analyzing {trading_pair}...")
                
                try:
                    # Analyze trading pair with custom lambda value for EWMA
                    results, price_data = analyze_trading_pair(
                        trading_pair, 
                        st.session_state.symbol_data, 
                        trading_pairs_config,
                        lambda_value=lambda_value
                    )
                    
                    # Store results if available
                    if results is not None:
                        st.session_state.all_results[trading_pair] = results
                        st.session_state.all_price_data[trading_pair] = price_data
                        
                        try:
                            # Calculate correlations
                            correlations = calculate_correlations(results)
                            st.session_state.all_correlations[trading_pair] = correlations
                            
                            # Calculate R²
                            r_squared = calculate_r_squared(results)
                            st.session_state.all_r_squared[trading_pair] = r_squared
                            
                            # Calculate optimal ranges with custom confidence level
                            ranges = analyze_optimal_ranges(
                                results, 
                                price_data, 
                                trading_pair, 
                                confidence=confidence_level
                            )
                            st.session_state.all_ranges[trading_pair] = ranges
                        except Exception as e:
                            st.warning(f"Warning: Secondary calculations for {trading_pair} failed: {str(e)}")
                    else:
                        st.warning(f"No valid results for {trading_pair}. Possibly insufficient data.")
                except Exception as e:
                    st.error(f"Error analyzing {trading_pair}: {str(e)}")
            
            # Step 3: Finalize
            status_container.info("Step 3/3: Preparing visualizations...")
            progress_bar.progress(0.9)  # 90% complete
            
            # Check if we have any valid results
            if not st.session_state.all_results:
                status_container.error("Analysis failed. No valid results were produced.")
                return
            
            # Finalize
            time.sleep(0.5)  # Short pause for visual feedback
            progress_bar.progress(1.0)
            status_container.success(f"Analysis complete! Analyzed {len(st.session_state.all_results)} trading pairs.")
            
            # Automatically collapse the progress section after completion
            time.sleep(1.5)
            st.rerun()  # Updated from st.experimental_rerun()
    
    # Run the analysis if button is clicked
    if run_analysis:
        run_volatility_analysis()
    
    # Display results if we have them
    if st.session_state.all_results:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Volatility Analysis", 
            "Correlation Analysis", 
            "Optimal Ranges", 
            "Raw Data",
            "Comparison View"  # New tab
        ])
        
        # Get all valid results
        valid_results = {pair: results for pair, results in st.session_state.all_results.items()
                         if results is not None and not results.empty}
        
        if not valid_results:
            st.warning("No valid results to display. Try analyzing different trading pairs.")
            return
        
        # VOLATILITY ANALYSIS TAB
        with tab1:
            st.markdown("<div class='section-header'>Volatility Analysis</div>", unsafe_allow_html=True)
            
            # Create a dropdown to select trading pair for detailed view
            selected_pair_for_vol = st.selectbox(
                "Select Trading Pair for Volatility Analysis",
                options=list(valid_results.keys()),
                key="vol_pair_selector"
            )
            
            if selected_pair_for_vol:
                # Use the custom volatility selector
                create_volatility_selector_dashboard(valid_results, selected_pair_for_vol)
                
                # Add expanded technical explanation with EWMA details
                with st.expander("Understanding Instantaneous Volatility and EWMA"):
                    st.markdown(expanded_volatility_explanation())
        
        # CORRELATION ANALYSIS TAB
        with tab2:
            st.markdown("<div class='section-header'>Correlation Analysis</div>", unsafe_allow_html=True)
            
            corr_tab1, corr_tab2 = st.tabs(["Correlation", "R-squared"])
            
            with corr_tab1:
            
                # Check if we have correlation data
                valid_correlations = {pair: corr for pair, corr in st.session_state.all_correlations.items() 
                                    if not corr.empty}
                
                if valid_correlations:
                    # Create two columns for the correlation visualizations
                    corr_col1, corr_col2 = st.columns([2, 1])
                    
                    with corr_col1:
                        # Create interactive correlation heatmap
                        try:
                            heatmap_fig = plot_correlation_heatmap_plotly(valid_correlations)
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating correlation heatmap: {str(e)}")
                    
                    with corr_col2:
                        # Create average correlations chart
                        try:
                            avg_corr_fig = plot_average_correlations_plotly(valid_correlations)
                            st.plotly_chart(avg_corr_fig, use_container_width=True)
                            
                            # Show the best lookback period
                            corr_df = pd.DataFrame(valid_correlations)
                            avg_correlations = corr_df.mean(axis=1, skipna=True)
                            non_nan_corr = avg_correlations.dropna()
                            
                            if not non_nan_corr.empty:
                                best_period = non_nan_corr.idxmax()
                                st.success(f"""
                                #### Best Lookback Period
                                **{best_period}** provides the best prediction of future volatility with an average correlation of **{non_nan_corr.max():.4f}**
                                """)
                        except Exception as e:
                            st.error(f"Error creating average correlations chart: {str(e)}")
                    
                    # Show correlation table in a cleaner format
                    with st.expander("View Correlation Table"):
                        corr_df = pd.DataFrame(valid_correlations)
                        
                        # Round values for cleaner display
                        formatted_corr_df = corr_df.round(4)
                        
                        # Display with highlights
                        st.dataframe(
                            formatted_corr_df.style.highlight_max(axis=0, color='#bce4d8').highlight_min(axis=0, color='#f7d1d1'),
                            use_container_width=True
                        )
                else:
                    st.warning("No valid correlation data available. This could be due to insufficient data points or missing future volatility values.")
                
                # Add explanation of correlation analysis
                with st.expander("About Correlation Analysis"):
                    st.markdown("""
                    **Interpreting the Correlation Analysis:**
                    
                    This analysis shows how well each lookback period correlates with future volatility:
                    
                    * **Higher positive correlation (closer to 1.0)** indicates that the lookback period is a better predictor of future volatility
                    * **Values near zero** indicate little predictive power
                    * **Negative values** suggest an inverse relationship
                    
                    The best lookback period for liquidity provisioning is typically the one with the highest correlation to future volatility, as it provides the most accurate estimation of the volatility you'll actually experience.
                    
                    **Note on EWMA:** While EWMA is excellent for estimating current instantaneous volatility, its correlation with future volatility can vary. In some market conditions, a fixed lookback period may better predict future movements.
                    """)
            with corr_tab2:
                st.markdown("### R² Analysis")
                st.markdown("""
                R² (coefficient of determination) measures how well the variance in future volatility is explained 
                by each lookback period. Higher values (closer to 1.0) indicate better predictive power.
                """)
                
                # Check if we have R² data
                valid_r_squared = {pair: r2 for pair, r2 in st.session_state.all_r_squared.items() 
                                if not r2.empty}
                
                if valid_r_squared:
                    # Create two columns for the R² visualizations
                    r2_col1, r2_col2 = st.columns([2, 1])
                    
                    with r2_col1:
                        # Create R² heatmap (similar to correlation heatmap)
                        try:
                            r2_df = pd.DataFrame(valid_r_squared).T
                            r2_fig = px.imshow(
                                r2_df,
                                color_continuous_scale='Viridis',
                                zmin=0,
                                zmax=1,
                                aspect="auto",
                                labels=dict(x="Trading Pair", y="Lookback Period", color="R²")
                            )
                            
                            r2_fig.update_layout(
                                title='R² of Lookback Volatilities with Future 1-day Volatility',
                                coloraxis_colorbar=dict(
                                    title="R²",
                                    thicknessmode="pixels", thickness=20,
                                    lenmode="pixels", len=300,
                                ),
                                margin=dict(l=40, r=40, t=80, b=40)
                            )
                            
                            # Add R² values as text
                            for i in range(len(r2_df.index)):
                                for j in range(len(r2_df.columns)):
                                    value = r2_df.iloc[i, j]
                                    if not np.isnan(value):
                                        text_color = 'white' if value > 0.5 else 'black'
                                        r2_fig.add_annotation(
                                            x=j, y=i,
                                            text=f"{value:.3f}",
                                            showarrow=False,
                                            font=dict(color=text_color)
                                        )
                            
                            st.plotly_chart(r2_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating R² heatmap: {str(e)}")
                    
                    with r2_col2:
                        # Create average R² chart
                        try:
                            avg_r2_fig = plot_r_squared_bar_chart(valid_r_squared)
                            st.plotly_chart(avg_r2_fig, use_container_width=True)
                            
                            # Show the best lookback period based on R²
                            r2_df = pd.DataFrame(valid_r_squared)
                            avg_r2 = r2_df.mean(axis=1, skipna=True)
                            non_nan_r2 = avg_r2.dropna()
                            
                            if not non_nan_r2.empty:
                                best_period = non_nan_r2.idxmax()
                                st.success(f"""
                                #### Best Lookback Period (R²)
                                **{best_period}** explains the most variance in future volatility with an average R² of **{non_nan_r2.max():.4f}**
                                """)
                        except Exception as e:
                            st.error(f"Error creating average R² chart: {str(e)}")
                    
                    # Show R² table in a cleaner format
                    with st.expander("View R² Table"):
                        r2_df = pd.DataFrame(valid_r_squared)
                        
                        # Round values for cleaner display
                        formatted_r2_df = r2_df.round(4)
                        
                        # Display with highlights
                        st.dataframe(
                            formatted_r2_df.style.highlight_max(axis=0, color='#bce4d8').highlight_min(axis=0, color='#f7d1d1'),
                            use_container_width=True
                        )
                else:
                    st.warning("No valid R² data available. This could be due to insufficient data points or missing future volatility values.")
                
                # Add explanation of R² analysis
                with st.expander("About R² Analysis"):
                    st.markdown("""
                    **Interpreting the R² Analysis:**
                    
                    R² (coefficient of determination) measures the proportion of variance in future volatility that is explained by each lookback period:
                    
                    * **R² = 1.0** means the lookback period perfectly explains future volatility
                    * **R² = 0.5** means the lookback period explains 50% of the variance in future volatility
                    * **R² = 0.0** means the lookback period has no explanatory power
                    
                    While correlation tells you about the direction and strength of the relationship, R² specifically tells you how much of the variability in future volatility can be predicted using each lookback period.
                    
                    **Why R² matters for trading:**
                    * Higher R² values indicate better predictive power
                    * R² helps you select the lookback period that will most reliably predict volatility
                    * The lookback period with the highest R² is typically the best choice for setting price ranges in Uniswap V3
                    """)
        
        # OPTIMAL RANGES TAB
        with tab3:
            st.markdown("<div class='section-header'>Optimal Price Ranges for Uniswap V3</div>", unsafe_allow_html=True)
            
            # Create a dropdown to select trading pair
            selected_pair_for_range = st.selectbox(
                "Select Trading Pair for Range Analysis",
                options=list(st.session_state.all_ranges.keys()),
                key="range_pair_selector"
            )
            
            if selected_pair_for_range:
                ranges = st.session_state.all_ranges[selected_pair_for_range]
                price_data = st.session_state.all_price_data[selected_pair_for_range]
                
                if ranges and price_data is not None and not price_data.empty:
                    # Get latest price
                    latest_price = price_data['price'].iloc[-1]
                    
                    # Create a table of optimal ranges
                    range_data = []
                    for model, data in ranges.items():
                        range_data.append({
                            "Model": model,
                            "Volatility": f"{data['volatility']:.4f}",
                            "Lower Bound": f"{data['lower_bound']:.6f}",
                            "Upper Bound": f"{data['upper_bound']:.6f}",
                            "Range Width": f"{data['range_width_pct']:.2f}%"
                        })
                    
                    if range_data:
                        range_df = pd.DataFrame(range_data)
                        st.dataframe(range_df, use_container_width=True)
                    
                    # Find the best lookback period from correlations
                    best_period = None
                    if selected_pair_for_range in st.session_state.all_correlations:
                        corr = st.session_state.all_correlations[selected_pair_for_range]
                        if not corr.empty:
                            best_period = corr.idxmax()
                    
                    # Show recommendation box
                    st.markdown("<div class='section-header'>Recommendation</div>", unsafe_allow_html=True)
                    
                    if best_period:
                        # Find the corresponding range
                        best_period_key = best_period.replace(' ', '_')
                        if best_period_key in ranges:
                            best_range = ranges[best_period_key]
                            
                            st.success(f"""
                            ### Recommended Range
                            
                            Based on correlation analysis, **{best_period}** is the best volatility predictor for {selected_pair_for_range}.
                            
                            **Range:** {best_range['lower_bound']:.6f} → {best_range['upper_bound']:.6f}
                            
                            **Width:** {best_range['range_width_pct']:.2f}%
                            
                            **Volatility estimate:** {best_range['volatility']:.4f}
                            """)
                        else:
                            st.info(f"Based on correlation analysis, the {best_period} lookback period has the highest predictive power, but no corresponding range data is available.")
                    else:
                        # If no best period found, suggest EWMA
                        if 'ewma' in ranges:
                            ewma_range = ranges['ewma']
                            
                            st.info(f"""
                            ### Suggested Range (EWMA)
                            
                            Using the EWMA model for instantaneous volatility estimation:
                            
                            **Range:** {ewma_range['lower_bound']:.6f} → {ewma_range['upper_bound']:.6f}
                            
                            **Width:** {ewma_range['range_width_pct']:.2f}%
                            
                            **Volatility estimate:** {ewma_range['volatility']:.4f}
                            """)
                else:
                    st.warning(f"No range data available for {selected_pair_for_range}")
            
            # Add explanation of optimal ranges
            with st.expander("About Optimal Price Ranges"):
                st.markdown("""
                **Understanding Optimal Price Ranges:**
                
                The optimal price range for Uniswap V3 liquidity provisioning depends on:
                
                1. **Volatility Estimation**: More accurate volatility estimation leads to better range setting
                2. **Confidence Level**: The ranges shown use a 95% confidence interval by default, meaning prices are expected to stay within this range 95% of the time
                3. **Trade-off**: Wider ranges capture more price movements but earn lower fees per dollar of liquidity
                4. **Best Practice**: Choose the volatility model with the highest correlation to future volatility
                
                For active management, the EWMA volatility model often provides the best real-time estimate of current market conditions, though historical windows (1 day, 3 days, 1 week, 2 weeks) may sometimes be better predictors of future volatility.
                """)
        
        # RAW DATA TAB
        with tab4:
            st.markdown("<div class='section-header'>Raw Data</div>", unsafe_allow_html=True)
            
            # Create a dropdown to select trading pair
            selected_pair_for_data = st.selectbox(
                "Select Trading Pair for Raw Data",
                options=list(st.session_state.all_results.keys()),
                key="data_pair_selector"
            )
            
            if selected_pair_for_data:
                results = st.session_state.all_results[selected_pair_for_data]
                price_data = st.session_state.all_price_data[selected_pair_for_data]
                
                # Create tabs for different data views
                data_tab1, data_tab2 = st.tabs(["Volatility Data", "Price Data"])
                
                with data_tab1:
                    if results is not None and not results.empty:
                        # Format for better readability
                        display_results = results.copy()
                        display_results.index = display_results.index.strftime('%Y-%m-%d')
                        
                        st.dataframe(display_results, use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv().encode('utf-8')
                        st.download_button(
                            label=f"Download {selected_pair_for_data} Volatility Data",
                            data=csv,
                            file_name=f"{selected_pair_for_data.replace('/', '_')}_volatility.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No volatility data available for {selected_pair_for_data}")
                
                with data_tab2:
                    if price_data is not None and not price_data.empty:
                        # Format for better readability
                        display_price = price_data.copy()
                        display_price.index = display_price.index.strftime('%Y-%m-%d')
                        
                        st.dataframe(display_price, use_container_width=True)
                        
                        # Download button
                        csv = price_data.to_csv().encode('utf-8')
                        st.download_button(
                            label=f"Download {selected_pair_for_data} Price Data",
                            data=csv,
                            file_name=f"{selected_pair_for_data.replace('/', '_')}_prices.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No price data available for {selected_pair_for_data}")
                        
        with tab5:
            st.markdown("<div class='section-header'>Multi-Pair Volatility Comparison</div>", unsafe_allow_html=True)
            
            if len(comparison_pairs) > 0:
                # Allow selecting which volatility measure to compare
                vol_measure = st.selectbox(
                    "Select Volatility Measure to Compare:",
                    options=["EWMA (Instantaneous)", "1 day", "3 days", "1 week"],
                    index=0
                )
                
                # Create mapping between display names and column names
                measure_mapping = {
                    "EWMA (Instantaneous)": "vol_ewma",
                    "1 day": "vol_1_day",
                    "3 days": "vol_3_days",
                    "1 week": "vol_1_week"
                }
                
                # Create figure for comparison
                fig = go.Figure()
                
                # Add traces for each selected pair
                for i, pair in enumerate(comparison_pairs):
                    if pair in valid_results:
                        results = valid_results[pair]
                        col_name = measure_mapping[vol_measure]
                        
                        if col_name in results.columns:
                            fig.add_trace(go.Scatter(
                                x=results.index,
                                y=results[col_name],
                                mode='lines',
                                name=pair,
                                line=dict(width=2),
                                hovertemplate='%{x}<br>%{y:.4f}<extra></extra>'
                            ))
                        else:
                            st.warning(f"Selected volatility measure not available for {pair}")
                    else:
                        st.warning(f"No data available for {pair}")
                
                # Update layout
                fig.update_layout(
                    title=f'Comparison of {vol_measure} Volatility Across Trading Pairs',
                    xaxis_title='Date',
                    yaxis_title='Annualized Volatility',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistical comparison
                if len(comparison_pairs) >= 2:
                    st.subheader("Statistical Comparison")
                    
                    stats_data = []
                    for pair in comparison_pairs:
                        if pair in valid_results and measure_mapping[vol_measure] in valid_results[pair].columns:
                            vol_data = valid_results[pair][measure_mapping[vol_measure]]
                            stats_data.append({
                                "Pair": pair,
                                "Mean Volatility": f"{vol_data.mean():.2f}",
                                "Max Volatility": f"{vol_data.max():.2f}",
                                "Min Volatility": f"{vol_data.min():.2f}",
                                "Volatility Std Dev": f"{vol_data.std():.2f}"
                            })
                    
                    if stats_data:
                        st.dataframe(pd.DataFrame(stats_data).set_index("Pair"), use_container_width=True)
            else:
                st.info("Please select at least one pair for comparison in the sidebar.")
        
    else:
        # Show welcome screen if no analysis has been run yet
        st.info("👈 Select trading pairs from the sidebar and click 'Run Analysis' to start.")
        
        # Add explanation of volatility for Uniswap V3
        st.markdown("""
        ## Volatility and Uniswap V3 Liquidity Provision
        
        In Uniswap V3, liquidity providers can concentrate their capital in specific price ranges, 
        unlike Uniswap V2 where liquidity is spread across the entire price curve.
        
        ### Why Instantaneous Volatility Matters
        
        1. **Concentrated Liquidity**: V3 allows concentration of capital in specific price ranges
        2. **Optimal Range Selection**: Requires accurate estimation of current and future volatility  
        3. **Traditional Volatility Methods Lag**: Fixed lookback windows (1 day, 1 week, etc.) don't adapt quickly to market changes

        ### This Tool Helps You:
        
        1. **Analyze historical volatility** using different lookback periods (1 day, 3 days, 1 week, 2 weeks)
        2. **Estimate instantaneous volatility** using EWMA methodology, which gives more weight to recent price movements
        3. **Compare volatility measures** to determine which best predicts future volatility
        4. **Calculate optimal price ranges** for liquidity provision based on volatility estimations
        
        Start by selecting trading pairs from the sidebar and running the analysis.
        
        ### Using EWMA for Instantaneous Volatility
        
        **EWMA (Exponentially Weighted Moving Average)** is particularly valuable because:
        
        * It gives more weight to recent price movements
        * It adapts quickly to volatility regime changes
        * It better captures the "right now" volatility
        * It's widely used in financial risk management (e.g., RiskMetrics methodology)
        
        The standard EWMA model uses λ=0.94, which you can adjust in the sidebar for more or less responsiveness to recent events.
        """)

def main():
    """
    Main function to run the dashboard or command-line analysis.
    """
    parser = argparse.ArgumentParser(description='Crypto Volatility Analysis')
    parser.add_argument('--dashboard', action='store_true', help='Run the Streamlit dashboard')
    parser.add_argument('--pairs', nargs='+', help='Trading pairs to analyze')
    parser.add_argument('--months', type=int, default=3, help='Months of historical data to retrieve')
    
    args = parser.parse_args()
    
    if args.dashboard or STREAMLIT_AVAILABLE:
        # Run the Streamlit dashboard
        run_streamlit_dashboard()
    else:
        # Run analysis directly (command-line mode)
        selected_pairs = args.pairs if args.pairs else ["SOL/USDC", "ETH/USDC"]
        
        # Initialize database connection
        global db_connection
        if db_connection is None:
            db_connection = setup_timescale_connection()
            
        if db_connection is None:
            print("Cannot run analysis: No database connection")
            return
            
        # Fetch symbol data
        symbol_data = fetch_all_symbols_data(symbols, months=args.months)
        
        # Analyze each trading pair
        all_results = {}
        all_correlations = {}
        
        for trading_pair in selected_pairs:
            print(f"\nAnalyzing {trading_pair}...")
            results, price_data = analyze_trading_pair(trading_pair, symbol_data, trading_pairs_config)
            
            if results is not None:
                all_results[trading_pair] = results
                
                # Calculate correlations
                correlations = calculate_correlations(results)
                all_correlations[trading_pair] = correlations
                
                print(f"Correlations for {trading_pair}:")
                print(correlations)
                
                # Analyze optimal ranges
                analyze_optimal_ranges(results, price_data, trading_pair)
        
        # Print overall findings
        if all_correlations:
            print("\nOverall findings:")
            avg_correlations = pd.DataFrame(all_correlations).mean(axis=1)
            print("Average correlations across all trading pairs:")
            print(avg_correlations)
            print(f"Best lookback period: {avg_correlations.idxmax()} (avg correlation: {avg_correlations.max():.4f})")
        else:
            print("\nNo valid results to analyze.")

if __name__ == "__main__":
    # Nothing needed here since we moved page_config to the run_streamlit_dashboard function
    
    # Run the main function
    main()
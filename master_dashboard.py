import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import math
import concurrent.futures
import time

# Set page title and layout
st.set_page_config(page_title="LP Strategy Optimizer", layout="wide")

# Title and description
st.title("Liquidity Provider Strategy Optimizer")
st.markdown("""
This dashboard helps you find the optimal range width for providing liquidity to DEX pairs.
It analyzes historical volatility and simulates LP positions to maximize returns while minimizing impermanent loss.
""")

# Global variables for database connection
db_connection = None

# ---------------- DATABASE CONNECTION FUNCTIONS ----------------

def setup_timescale_connection():
    """
    Set up connection to Timescale database.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get database credentials from environment variables
    db_host = os.getenv("TIMESCALE_HOST")
    db_port = os.getenv("TIMESCALE_PORT", "34569")
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
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return None
    
    try:
        # Connect to the database with SSL enabled
        connection = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password,
            sslmode='require',
            connect_timeout=10
        )
        st.success("Successfully connected to database")
        return connection
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        st.info("Ensure your .env file contains TIMESCALE_HOST, TIMESCALE_PORT, TIMESCALE_DATABASE, TIMESCALE_USER, and TIMESCALE_PASSWORD")
        return None

def fetch_trading_pair_directly(connection, trading_pair, hours=24):
    """
    Fetch price history for a trading pair directly from Timescale DB.
    """
    if connection is None:
        st.error(f"Cannot fetch data for {trading_pair}: No database connection")
        return None
    
    # Calculate the start date
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    # Remove the slash for database querying if necessary
    db_pair_format = trading_pair.replace('/', '')
    
    st.info(f"Fetching data for {trading_pair} from {start_date} to {end_date}...")
    
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
            
            # Close cursor and create a new one to avoid any transaction issues
            cursor.close()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Identify the correct column names for datetime and close
            date_column = next((col for col in columns if 'time' in col.lower() or 'date' in col.lower()), None)
            close_column = next((col for col in columns if 'close' in col.lower()), None)
            
            if not date_column or not close_column:
                st.error(f"Could not identify required columns. Available columns: {columns}")
                cursor.close()
                return None
                
            # Query the database for the price history, trying first with the formatted pair
            query = f"""
                SELECT {date_column} as datetime, {close_column} as price
                FROM public.all_crypto_prices
                WHERE symbol = %s
                AND {date_column} >= %s
                AND {date_column} <= %s
                ORDER BY {date_column}
            """
            
            # Try with the original trading pair format first
            cursor.execute(query, (trading_pair, start_date.isoformat(), end_date.isoformat()))
            records = cursor.fetchall()
            
            # If no records, try with the slash removed
            if not records:
                cursor.execute(query, (db_pair_format, start_date.isoformat(), end_date.isoformat()))
                records = cursor.fetchall()
            
            # If still no records, try other common formats
            if not records:
                # Try with underscore instead of slash
                cursor.execute(query, (trading_pair.replace('/', '_'), start_date.isoformat(), end_date.isoformat()))
                records = cursor.fetchall()
            
            # Close the cursor
            cursor.close()
            
            # If still no records, check what symbols are available
            if not records:
                cursor = connection.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM public.all_crypto_prices LIMIT 100")
                available_symbols = [row[0] for row in cursor.fetchall()]
                cursor.close()
                
                st.warning(f"No data found for {trading_pair}. Available symbols: {available_symbols[:20]}...")
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(records)
            
            if df.empty:
                st.warning(f"No data found for symbol {trading_pair}")
                return None
                
            # Convert datetime to proper format and set as index
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Keep only unique datetimes by taking the last value for each
            df = df.drop_duplicates(subset=['datetime'], keep='last')
            df = df.set_index('datetime')
            
            # Sort by datetime to ensure data is in chronological order
            df = df.sort_index()
            
            # Resample to 15-minute intervals if necessary
            if len(df) > 0:
                first_interval = df.index[1] - df.index[0] if len(df) > 1 else timedelta(minutes=15)
                if first_interval.total_seconds() < 60*10:  # If intervals are less than 10 minutes
                    st.info(f"Resampling data to 15-minute intervals (original interval: {first_interval})...")
                    df = df.resample('15T').last().dropna()
            
            # Check for sufficient data
            if len(df) < 10:  # Arbitrary threshold
                st.warning(f"Insufficient data for {trading_pair}: only {len(df)} data points found")
                return None
            
            st.success(f"Retrieved {len(df)} data points for {trading_pair}")
            return df
            
        except Exception as inner_e:
            st.error(f"Error in data processing for {trading_pair}: {str(inner_e)}")
            cursor.close()
            connection.rollback()  # Reset any failed transaction
            return None
    
    except Exception as e:
        st.error(f"Error fetching data for {trading_pair}: {str(e)}")
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        connection.rollback()  # Reset the failed transaction
        return None

def calculate_pair_price(symbol_data, numerator_symbol, denominator_symbol):
    """
    Calculate price for a trading pair from individual symbol prices.
    """
    # Get dataframes for both symbols
    numerator_df = symbol_data.get(numerator_symbol)
    denominator_df = symbol_data.get(denominator_symbol)
    
    # USDC special handling - if USDC is missing, create a synthetic dataframe with value 1.0
    if denominator_symbol == "USDC" and (denominator_df is None or denominator_df.empty):
        st.info(f"Creating synthetic USDC data as fallback (value = 1.0)")
        
        # If we have numerator data, use its index for USDC
        if numerator_df is not None and not numerator_df.empty:
            # Create a DataFrame with same index as numerator but all values = 1.0
            denominator_df = pd.DataFrame(index=numerator_df.index, data={'close': 1.0})
            # Add to symbol_data so it can be reused
            symbol_data[denominator_symbol] = denominator_df
        else:
            st.error(f"Cannot create synthetic USDC data because {numerator_symbol} data is missing")
            return None
    
    if numerator_df is None or denominator_df is None:
        st.error(f"Missing data for {numerator_symbol} or {denominator_symbol}")
        return None
    
    # Merge dataframes on datetime index
    merged_df = pd.merge(
        numerator_df, 
        denominator_df, 
        left_index=True, 
        right_index=True,
        suffixes=(f'_{numerator_symbol}', f'_{denominator_symbol}')
    )
    
    if merged_df.empty:
        st.warning(f"No overlapping data points for {numerator_symbol}/{denominator_symbol}")
        return None
    
    # Calculate pair price
    merged_df['price'] = merged_df[f'close_{numerator_symbol}'] / merged_df[f'close_{denominator_symbol}']
    
    # Remove any rows with infinite or NaN values
    merged_df = merged_df[np.isfinite(merged_df['price'])]
    
    if merged_df.empty:
        st.warning(f"No valid price data for {numerator_symbol}/{denominator_symbol} after filtering")
        return None
    
    # Ensure data is sorted by datetime
    merged_df = merged_df.sort_index()
    
    # Only keep the price column
    result_df = pd.DataFrame(merged_df['price'])
    
    # Drop any remaining NaN values
    result_df = result_df.dropna()
    
    st.success(f"Calculated {len(result_df)} price points for {numerator_symbol}/{denominator_symbol}")
    
    return result_df

def fetch_symbol_data(symbol, hours=24, connection=None):
    """
    Fetch price history for a single symbol from Timescale DB.
    For backward compatibility only - prefer fetch_trading_pair_directly.
    """
    st.warning(f"Using legacy fetch_symbol_data method for {symbol}. This may not work with your current database schema.")
    
    # Make sure we have a database connection
    if connection is None:
        global db_connection
        if db_connection is None:
            db_connection = setup_timescale_connection()
        connection = db_connection
    
    if connection is None:
        st.error(f"Cannot fetch data for {symbol}: No database connection")
        return None
    
    # Calculate the start date
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    
    st.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Try a simple query to get available data
        cursor.execute("""
            SELECT * FROM public.all_crypto_prices 
            WHERE symbol = %s 
            AND datetime >= %s 
            AND datetime <= %s
            ORDER BY datetime
            LIMIT 100
        """, (symbol, start_date.isoformat(), end_date.isoformat()))
        
        records = cursor.fetchall()
        cursor.close()
        
        if not records:
            return None
            
        df = pd.DataFrame(records)
        
        if 'close' in df.columns:
            price_col = 'close'
        elif 'price' in df.columns:
            price_col = 'price'
        else:
            # Find a numeric column that might be price
            price_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
            if price_cols:
                price_col = price_cols[0]
            else:
                return None
        
        # Create a simplified dataframe
        result_df = pd.DataFrame({
            'datetime': pd.to_datetime(df['datetime']),
            'close': df[price_col]
        })
        
        result_df = result_df.set_index('datetime')
        return result_df
        
    except Exception as e:
        st.error(f"Error in legacy fetch_symbol_data for {symbol}: {str(e)}")
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        return None

def fetch_price_data(connection, trading_pair, hours=24):
    """
    Fetch price history for a trading pair from the Timescale database.
    """
    try:
        # Try to fetch trading pair data directly
        price_data = fetch_trading_pair_directly(connection, trading_pair, hours)
        
        # If successful, return the data
        if price_data is not None and not price_data.empty:
            return price_data
        
        # If direct approach fails, try the old method of fetching individual symbols
        st.warning(f"Failed to fetch {trading_pair} directly. Trying to fetch individual symbols...")
        
        # Parse the trading pair to get base and quote currencies
        base_currency, quote_currency = trading_pair.split('/')
        
        # Fetch data for both symbols
        symbol_data = {}
        for symbol in [base_currency, quote_currency]:
            symbol_data[symbol] = fetch_symbol_data(symbol, hours, connection)
        
        # Calculate the trading pair price
        price_data = calculate_pair_price(symbol_data, base_currency, quote_currency)
        
        return price_data
        
    except Exception as e:
        st.error(f"Error fetching price data: {str(e)}")
        
        # For demo/development purposes - generate synthetic data if fetch fails
        if st.checkbox("Use synthetic data for testing?"):
            st.warning("Using synthetic price data. Not for production use!")
            return generate_synthetic_price_data(trading_pair, hours)
            
        return None

def generate_synthetic_price_data(trading_pair, hours=24):
    """
    Generate synthetic price data for testing purposes.
    """
    # Set seed based on trading pair for reproducibility
    np.random.seed(hash(trading_pair) % 10000)
    
    # Generate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=hours*4)  # 15-minute intervals
    
    # Generate price data with realistic properties
    base_price = 100.0
    if 'BTC' in trading_pair:
        base_price = 30000.0
    elif 'ETH' in trading_pair:
        base_price = 2000.0
    elif 'SOL' in trading_pair and 'BTC' not in trading_pair and 'ETH' not in trading_pair:
        base_price = 50.0
    elif 'SUI' in trading_pair:
        base_price = 1.5
    elif 'FART' in trading_pair:
        base_price = 0.01
    
    # Add realistic volatility
    volatility = 0.01  # 1% hourly volatility
    
    # Generate log returns
    returns = np.random.normal(0, volatility, len(timestamps))
    
    # Convert to prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices
    }, index=timestamps)
    
    return df

# ---------------- VOLATILITY CALCULATION FUNCTIONS ----------------

def calculate_returns(prices):
    """
    Calculate logarithmic returns from price data.
    """
    # Remove any zero or negative values
    prices = prices[prices > 0]
    
    if len(prices) < 2:
        st.warning("Not enough price data to calculate returns")
        return pd.Series()
    
    # Calculate log returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    return returns

def calculate_volatility(returns, window_hours, annualize=True, interval_minutes=15):
    """
    Calculate volatility from returns using a fixed lookback window.
    """
    if returns.empty or len(returns) < 2:
        return None
    
    # Calculate window size in number of data points
    window_size = int(window_hours * (60 / interval_minutes))
    
    # Ensure we have enough data for the window
    if len(returns) < window_size:
        st.warning(f"Not enough data for {window_hours} hour window. Using available data.")
        window_size = len(returns)
    
    # Calculate standard deviation of returns over the window
    vol = returns.iloc[-window_size:].std()
    
    # Annualize by multiplying by sqrt(intervals in a year)
    if annualize:
        # For 15-minute intervals: 4 intervals per hour * 24 hours * 365 days
        intervals_per_year = (60 / interval_minutes) * 24 * 365
        vol = vol * np.sqrt(intervals_per_year)
    
    return vol

def calculate_ewma_volatility(returns, lambda_=0.94, annualize=True, interval_minutes=15):
    """
    Calculate EWMA (Exponentially Weighted Moving Average) volatility.
    """
    if returns.empty or len(returns) < 2:
        return None
    
    # Square the returns
    returns_squared = returns ** 2
    
    # Calculate EWMA of squared returns
    ewma = returns_squared.ewm(alpha=(1-lambda_)).mean().iloc[-1]
    
    # Take the square root to get volatility
    vol = np.sqrt(ewma)
    
    # Annualize
    if annualize:
        # For 15-minute intervals: 4 intervals per hour * 24 hours * 365 days
        intervals_per_year = (60 / interval_minutes) * 24 * 365
        vol = vol * np.sqrt(intervals_per_year)
    
    return vol

# ---------------- LP SIMULATION FUNCTIONS ----------------

def return_price_path(T, N, dt, mu, sigma):
    """
    Generate a simulated price path using Geometric Brownian Motion.
    """
    # Initial condition
    S0 = 100  # Starting price of the asset
    
    # Random component: generate all changes at once
    Z = np.random.normal(0, 1, N)
    
    # Calculate the percentage changes in price
    percentage_changes = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Calculate the price path using the cumulative product of changes
    price_path = np.zeros(N+1)
    price_path[0] = S0
    for i in range(1, N+1):
        price_path[i] = price_path[i-1] * np.exp(percentage_changes[i-1])

    # Convert the arrays into dictionaries
    price_path_dict = {i: price_path[i] for i in range(N+1)}
    
    return price_path_dict

def calculate_L(V, P, t):
    """
    Calculate constant for LP position.
    """
    return V/((2*np.sqrt(P)-(P + P) / (np.sqrt(P) * (1.0001 ** t))))

def calculate_V(P, P0, L, t):
    """
    Calculate value of LP position.
    """
    return L*(2*np.sqrt(P)-(P + P0) / (np.sqrt(P0) * (1.0001 ** t)))

def rebalance_lp_position(price, lp_position):
    """
    Rebalance LP position when price moves outside the specified range.
    Returns the updated position and whether a rebalance occurred.
    """
    rebalanced = False
    
    if (price < lp_position['p_lower_rebalance'] or 
        price > lp_position['p_upper_rebalance']):
        
        # Apply swap fee, update range and recalculate constants
        lp_position['value'] = lp_position['value'] * (1 - (0.5 - lp_position['imbalance']) * lp_position['fee_rate'])
        lp_position['p_0'] = price
        lp_position['p_lower'] = lp_position['p_0'] * (1.0001 ** (-lp_position['num_ticks']))
        lp_position['p_upper'] = lp_position['p_0'] * (1.0001 ** (lp_position['num_ticks']))
        lp_position['p_lower_rebalance'] = lp_position['p_lower'] * (1.0001 ** (lp_position['num_ticks'] * lp_position['imbalance'] * 2))
        lp_position['p_upper_rebalance'] = lp_position['p_upper'] * (1.0001 ** (-lp_position['num_ticks'] * lp_position['imbalance'] * 2))
        lp_position['L'] = calculate_L(lp_position['value'], lp_position['p_0'], lp_position['num_ticks'])
        rebalanced = True
        
    return lp_position, rebalanced

def simulate_lp_value(price_path, V, t, imbalance, fee_rate):
    """
    Simulate LP value over time based on a price path.
    Returns the simulation results and the number of rebalances.
    """
    lp_position = {}
    lp_position['value'] = V
    lp_position['price'] = price_path[0]
    lp_position['p_0'] = price_path[0]
    lp_position['num_ticks'] = t 
    lp_position['p_lower'] = lp_position['p_0'] * (1.0001 ** (-t))
    lp_position['p_upper'] = lp_position['p_0'] * (1.0001 ** (t))
    lp_position['L'] = calculate_L(lp_position['value'], lp_position['p_0'], lp_position['num_ticks'])
    lp_position['p_lower_rebalance'] = lp_position['p_lower'] * (1.0001 ** (t * imbalance * 2))
    lp_position['p_upper_rebalance'] = lp_position['p_upper'] * (1.0001 ** (-t * imbalance * 2))
    lp_position['imbalance'] = imbalance
    lp_position['fee_rate'] = fee_rate
    
    simulation = pd.DataFrame([lp_position])
    rebalance_count = 0
    
    for time_step in price_path:
        lp_position['value'] = calculate_V(price_path[time_step], 
                                           lp_position['p_0'], 
                                           lp_position['L'], 
                                           lp_position['num_ticks'])
        
        lp_position['price'] = price_path[time_step]
        
        lp_position, rebalanced = rebalance_lp_position(price_path[time_step], lp_position)
        if rebalanced:
            rebalance_count += 1
        
        new_row = pd.DataFrame([lp_position])
        simulation = pd.concat([simulation, new_row], ignore_index=True)
    
    return simulation, rebalance_count

def simulate_single_path(path_index, price_path_params, tick_spacings, initial_value, imbalance, fee_rate, N):
    """
    Helper function to simulate a single price path for all range widths.
    This function is designed to be used with parallel processing.
    """
    # Generate price path
    T, mu, sigma = price_path_params
    dt = T / N
    price_path = return_price_path(T, N, dt, mu, sigma)
    
    # Initialize results for this path
    results = np.zeros(len(tick_spacings))
    rebalance_counts = np.zeros(len(tick_spacings))
    
    # Run simulation for each tick spacing
    for j, spacing in enumerate(tick_spacings):
        sim_result, rebalance_count = simulate_lp_value(price_path, initial_value, spacing, imbalance, fee_rate)
        
        # Store change in liquidity constant (L) as a percentage
        results[j] = 100 * (sim_result.at[N, 'L']/sim_result.at[0, 'L'] - 1)
        
        # Store rebalance count
        rebalance_counts[j] = rebalance_count
    
    return results, rebalance_counts

def run_lp_simulations(sigma, num_price_paths=10, initial_value=1000, 
                     imbalance=0.10, fee_rate=0.00, projected_daily_fee_pct=0.04):
    """
    Run multiple LP simulations with fixed percentage-based range widths.
    Uses parallel processing to speed up simulations.
    """
    # Set simulation parameters
    T = 1.0/365  # 1 day
    N = 1440     # Number of time steps (1-minute intervals)
    dt = T / N   # Time step
    mu = 0.00    # Expected annual return (drift)
    price_path_params = (T, mu, sigma)
    
    # Define the range widths in percentage (one-sided)
    # These will be +/- the percentage (so the total width is 2x these values)
    range_widths_pct = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]
    
    # Convert percentage widths to tick spacings
    # Formula: ticks = log(1 + width_pct/100) / log(1.0001)
    tick_spacings = [round(math.log(1 + width_pct/100) / math.log(1.0001)) for width_pct in range_widths_pct]
    
    # Array to store results
    num_widths = len(range_widths_pct)
    results = np.zeros((num_price_paths, num_widths))
    rebalance_counts = np.zeros((num_price_paths, num_widths))
    
    with st.spinner(f"Running {num_price_paths} simulations for {num_widths} different range widths..."):
        # Show time estimate
        start_time = time.time()
        
        progress_bar = st.progress(0)
        
        # Run simulations in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit tasks for each price path
            futures = [
                executor.submit(
                    simulate_single_path, 
                    i, 
                    price_path_params, 
                    tick_spacings, 
                    initial_value, 
                    imbalance, 
                    fee_rate, 
                    N
                ) for i in range(num_price_paths)
            ]
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                path_results, path_rebalances = future.result()
                results[i] = path_results
                rebalance_counts[i] = path_rebalances
                
                # Update progress
                progress_bar.progress((i + 1) / num_price_paths)
        
        # Show time taken
        end_time = time.time()
        st.success(f"Simulations completed in {end_time - start_time:.2f} seconds")
    
    # Calculate average results
    avg_results = np.mean(results, axis=0)
    avg_rebalances = np.mean(rebalance_counts, axis=0)
    
    # Calculate total width in percentage terms (double the one-sided width)
    total_width_pcts = [width_pct * 2 for width_pct in range_widths_pct]
    
    # Calculate projected fees based on width
    # Narrower ranges earn more fees (inverse relationship)
    # Base projection is for +/-1% range (2% total width)
    base_width_pct = 2.0
    projected_fees = [projected_daily_fee_pct * 100 * (base_width_pct / width_pct) for width_pct in total_width_pcts]
    
    # Calculate net returns (IL from simulations + projected fees)
    net_returns = [avg_results[i] + projected_fees[i] for i in range(num_widths)]
    
    # Find optimal tick spacing
    optimal_idx = np.argmax(net_returns)
    optimal_tick_spacing = tick_spacings[optimal_idx]
    optimal_width_pct = range_widths_pct[optimal_idx]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Range Width (±%)': range_widths_pct,
        'Total Width (%)': total_width_pcts,
        'Tick Spacing': tick_spacings,
        'Impermanent Loss (%)': avg_results,
        'Projected Fees (%)': projected_fees,
        'Net Return (%)': net_returns,
        'Expected Daily Rebalances': avg_rebalances
    })
    
    return results_df, optimal_width_pct, optimal_tick_spacing

# ---------------- MAIN APPLICATION LOGIC ----------------

# Initialize connection to the database
db_connection = setup_timescale_connection()

# Define trading pairs
solana_pairs = [
    "SOL/USDC", "FART/USDC", "CBBTC/USDC", 
    "FART/SOL", "SPX/SOL", "SPX/USDC", "CBBTC/SOL"
]

sui_pairs = [
    "SUI/USDC", "WAL/USDC", "WAL/SUI", 
    "DEEP/SUI", "DEEP/USDC", "WBTC/SUI"
]

# UI layout
st.sidebar.header("Configuration")

# Blockchain selection
blockchain = st.sidebar.radio("Select Blockchain", ["Solana", "Sui"])

# Trading pair selection
if blockchain == "Solana":
    trading_pair = st.sidebar.selectbox("Select Trading Pair", solana_pairs)
else:
    trading_pair = st.sidebar.selectbox("Select Trading Pair", sui_pairs)

# Data settings
st.sidebar.subheader("Data Settings")
hours = st.sidebar.slider("Hours of Historical Data", 24, 168, 24)
use_synthetic = st.sidebar.checkbox("Use Synthetic Data", value=False)

# Volatility settings
st.sidebar.subheader("Volatility Settings")
vol_method = st.sidebar.radio(
    "Volatility Calculation Method",
    ["12-Hour", "24-Hour", "EWMA"]
)

# Simulation settings
st.sidebar.subheader("Simulation Settings")
num_price_paths = st.sidebar.number_input("Number of Price Paths", 5, 100, 10)
initial_value = st.sidebar.number_input("Initial LP Value", 100, 10000, 1000)
imbalance = st.sidebar.number_input("Imbalance Threshold", 0.01, 0.50, 0.10, format="%.2f")
fee_rate = st.sidebar.number_input("Swap Fee Rate", 0.0, 0.01, 0.0, format="%.4f")

# Explain projected fees
st.sidebar.subheader("Fee Projections")
st.sidebar.markdown("""
**Projected Daily Fee %** is the estimated daily fee percentage for a +/-1% range (2% liquidity depth).
This is similar to the APR estimate shown by DEXs like Orca when providing liquidity.
""")
projected_fee_pct = st.sidebar.number_input("Projected Daily Fee %", 0.01, 0.20, 0.04, format="%.2f")

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Display information about the selected pair
st.header(f"Analysis for {trading_pair}")

# Fetch historical price data if button is clicked
if run_analysis:
    with st.spinner("Fetching price data..."):
        if use_synthetic:
            price_data = generate_synthetic_price_data(trading_pair, hours)
            st.info("Using synthetic price data for demonstration")
        else:
            price_data = fetch_price_data(db_connection, trading_pair, hours)
        
        if price_data is None or price_data.empty:
            st.error("Failed to retrieve price data. Please try another pair or use synthetic data.")
            st.stop()
    
    # Display price chart using Plotly
    st.subheader("Historical Price Data")
    
    fig = px.line(
        price_data.reset_index(), 
        x='datetime', 
        y='price',
        title=f"{trading_pair} Price - Last {hours} Hours"
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate returns and volatility
    returns = calculate_returns(price_data['price'])
    
    if returns.empty:
        st.error("Could not calculate returns from price data.")
        st.stop()
    
    # Calculate volatilities
    vol_12h = calculate_volatility(returns, 12)
    vol_24h = calculate_volatility(returns, 24)
    vol_ewma = calculate_ewma_volatility(returns)
    
    # Display volatility metrics
    st.subheader("Volatility Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("12-Hour Volatility", f"{vol_12h:.2%}", help="Annualized volatility based on 12-hour window")
    
    with col2:
        st.metric("24-Hour Volatility", f"{vol_24h:.2%}", help="Annualized volatility based on 24-hour window")
    
    with col3:
        st.metric("EWMA Volatility", f"{vol_ewma:.2%}", help="Exponentially Weighted Moving Average volatility")
    
    # Select appropriate volatility for simulation
    if vol_method == "12-Hour":
        selected_vol = vol_12h
        st.info(f"Using 12-Hour volatility ({selected_vol:.2%}) for simulations")
    elif vol_method == "24-Hour":
        selected_vol = vol_24h
        st.info(f"Using 24-Hour volatility ({selected_vol:.2%}) for simulations")
    else:  # EWMA
        selected_vol = vol_ewma
        st.info(f"Using EWMA volatility ({selected_vol:.2%}) for simulations")
    
    # Run LP simulations
    results_df, optimal_width_pct, optimal_tick_spacing = run_lp_simulations(
        sigma=selected_vol,
        num_price_paths=num_price_paths,
        initial_value=initial_value,
        imbalance=imbalance,
        fee_rate=fee_rate,
        projected_daily_fee_pct=projected_fee_pct
    )
    
    # Display results
    st.subheader("Simulation Results")
    
    # Plot results using Plotly
    st.write("LP Performance vs Range Width")
    
    # Create Plotly figure with subplots
    fig = go.Figure()
    
    # Create a secondary y-axis for rebalances
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=results_df['Total Width (%)'],
        y=results_df['Impermanent Loss (%)'],
        mode='lines',
        name='Impermanent Loss',
        line=dict(color='red', dash='dash')
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=results_df['Total Width (%)'],
        y=results_df['Projected Fees (%)'],
        mode='lines',
        name='Projected Fees',
        line=dict(color='green', dash='dash')
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=results_df['Total Width (%)'],
        y=results_df['Net Return (%)'],
        mode='lines',
        name='Net Return',
        line=dict(color='blue', width=3)
    ), secondary_y=False)
    
    # Add rebalances on secondary y-axis
    fig.add_trace(go.Scatter(
        x=results_df['Total Width (%)'],
        y=results_df['Expected Daily Rebalances'],
        mode='lines+markers',
        name='Expected Rebalances',
        line=dict(color='purple', dash='dot'),
        marker=dict(size=7)
    ), secondary_y=True)
    
    # Highlight optimal point
    optimal_row = results_df[results_df['Range Width (±%)'] == optimal_width_pct].iloc[0]
    optimal_width = optimal_row['Total Width (%)']
    optimal_return = optimal_row['Net Return (%)']
    
    fig.add_trace(go.Scatter(
        x=[optimal_width],
        y=[optimal_return],
        mode='markers',
        name=f'Optimal: ±{optimal_width_pct}% range',
        marker=dict(color='green', size=15),
        hoverinfo='text',
        hovertext=f'Optimal Width: ±{optimal_width_pct}%<br>Total Width: {optimal_width}%<br>Return: {optimal_return:.2f}%<br>Rebalances: {optimal_row["Expected Daily Rebalances"]:.1f}/day'
    ), secondary_y=False)
    
    # Update layout
    fig.update_layout(
        title=f'LP Performance Analysis for {trading_pair}',
        xaxis_title='Total Range Width (%)',
        height=600,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Return (%)", secondary_y=False)
    fig.update_yaxes(title_text="Daily Rebalances", secondary_y=True)
    
    # Add a note explaining the chart
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Wider ranges have less impermanent loss but earn fewer fees.<br>Narrower ranges earn more fees but risk more impermanent loss and frequent rebalancing.",
        showarrow=False,
        font=dict(size=12),
        align="center",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display optimal strategy
    st.subheader("Optimal LP Strategy")
    
    optimal_row = results_df[results_df['Range Width (±%)'] == optimal_width_pct].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Optimal Range Width", f"±{optimal_width_pct:.2f}%")
    
    with col2:
        st.metric("Tick Spacing", f"{int(optimal_row['Tick Spacing'])}")
    
    with col3:
        st.metric("Projected Daily Return", f"{optimal_row['Net Return (%)']:.2f}%")
        
    with col4:
        st.metric("Expected Rebalances", f"{optimal_row['Expected Daily Rebalances']:.1f}/day")
    
    # Display lower and upper price bounds
    current_price = price_data['price'].iloc[-1]
    
    # Calculate price bounds based on the percentage range
    lower_bound = current_price * (1 - optimal_width_pct/100)
    upper_bound = current_price * (1 + optimal_width_pct/100)
    
    st.write(f"Based on the current price of {current_price:.6f}, your LP position should be set with:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Lower Price Bound", f"{lower_bound:.6f}")
    
    with col2:
        st.metric("Upper Price Bound", f"{upper_bound:.6f}")
    
    # Display detailed results
    with st.expander("Detailed Simulation Results"):
        st.dataframe(results_df)
else:
    st.info("Click 'Run Analysis' to start the simulation and find the optimal LP range.")

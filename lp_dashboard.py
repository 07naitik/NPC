import streamlit as st
import numpy as np
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
from functools import partial
import altair as alt
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Liquidity Provider (LP) Simulation Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stAlert {
        background-color: #E0E7FF;
        border: 1px solid #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🚀 Advanced LP Position Simulator")
st.markdown("""
This dashboard simulates and analyzes Liquidity Provider (LP) positions in concentrated liquidity protocols like Uniswap V3.
Discover optimal strategies for different market conditions and compare performance metrics.
""")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Market parameters
st.sidebar.subheader("Market Parameters")
time_period = st.sidebar.selectbox(
    "Time Period",
    ["1 Day", "1 Week", "1 Month"],
    help="The time period to simulate"
)

time_mapping = {
    "1 Day": 1.0/365,
    "1 Week": 7.0/365,
    "1 Month": 30.0/365
}

T = time_mapping[time_period]
N = st.sidebar.slider("Simulation Steps", 100, 1440, 1440, 
                      help="Number of time steps in the simulation (1440 = one per minute in a day)")
dt = T / N

sigma = st.sidebar.slider("Market Volatility (Annual)", 0.1, 2.0, 1.0, 0.1, 
                         help="Higher values mean more price movement")
mu = st.sidebar.slider("Market Drift (Annual Return %)", -20.0, 20.0, 0.0, 1.0, 
                      help="Expected annual return (positive = uptrend, negative = downtrend)")
mu = mu / 100  # Convert percentage to decimal

# LP position parameters
st.sidebar.subheader("LP Position Parameters")
LP_initial_value = st.sidebar.number_input("Initial LP Value ($)", 100, 10000, 1000, 100,
                                         help="Starting capital in your LP position")
tick_spacing = st.sidebar.slider("Tick Spacing", 1, 50, 10, 
                               help="Base unit for position width (higher = wider positions)")
imbalance = st.sidebar.slider("Rebalance Threshold (%)", 1.0, 50.0, 10.0, 1.0,
                            help="Percentage imbalance required to trigger a rebalance") / 100

fee_tier = st.sidebar.selectbox(
    "Fee Tier",
    ["0.01% (Very Low)", "0.05% (Low)", "0.3% (Medium)", "1% (High)"],
    index=2,
    help="Trading fee percentage charged on swaps"
)

fee_map = {
    "0.01% (Very Low)": 0.0001,
    "0.05% (Low)": 0.0005,
    "0.3% (Medium)": 0.003,
    "1% (High)": 0.01
}
fee_rate = fee_map[fee_tier]

rebalance_cost = st.sidebar.slider("Rebalance Cost (Gas) in $", 0.0, 10.0, 2.0, 0.1,
                                 help="Transaction cost for each rebalance")

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
num_price_path = st.sidebar.slider("Number of Simulations", 1, 100, 10, 
                                  help="More simulations = more reliable results but slower")
num_tick_spacing = st.sidebar.slider("Number of LP Strategies", 5, 50, 20,
                                    help="Number of different position widths to compare")

# Advanced options toggle
show_advanced = st.sidebar.checkbox("Show Advanced Options")
if show_advanced:
    st.sidebar.subheader("Advanced Parameters")
    starting_price = st.sidebar.number_input("Starting Price ($)", 1.0, 10000.0, 100.0, 1.0,
                                           help="Initial price of the asset")
    projected_fees_annual = st.sidebar.slider("Projected Annual Fee APY (%)", 0.1, 100.0, 4.0, 0.1,
                                            help="Estimated annual percentage yield from fees") / 100
    projected_fees = projected_fees_annual / 365 * (T * 365)  # Convert to simulation timeframe
    enable_multiprocessing = st.sidebar.checkbox("Enable Multiprocessing", True,
                                               help="Use multiple CPU cores to speed up simulation")
else:
    starting_price = 100.0
    projected_fees_annual = 0.04  # 4% annual
    projected_fees = projected_fees_annual / 365 * (T * 365)  # Convert to simulation timeframe
    enable_multiprocessing = True

# Button to trigger simulation
run_simulation = st.sidebar.button("Run Simulation", type="primary")

# Simulation functions with improvements
@st.cache_data
def return_price_path(T, N, dt, mu, sigma, starting_price=100.0, seed=None):
    """
    Generate a price path using geometric Brownian motion.
    
    Parameters:
    - T: Time period in years
    - N: Number of time steps
    - dt: Time step size
    - mu: Expected annual return (drift)
    - sigma: Annualized volatility
    - starting_price: Initial price of the asset
    - seed: Random seed for reproducibility
    
    Returns:
    - Dictionary mapping time steps to prices
    - Numpy array of prices
    """
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    # Initial condition
    S0 = starting_price
    
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
    
    return price_path_dict, price_path

# Improved LP position calculations with better documentation
def calculate_L(V, P, t):
    """
    Calculate the liquidity constant (L) for a given LP position.
    
    Parameters:
    - V: Value of the LP position
    - P: Current price
    - t: Number of ticks (position width)
    
    Returns:
    - L: Liquidity constant
    """
    try:
        # Apply the mathematical formula from concentrated liquidity pools
        denominator = 2*np.sqrt(P) - (P + P) / (np.sqrt(P) * (1.0001 ** t))
        if denominator <= 0:
            return 0  # Invalid position
        return V / denominator
    except Exception as e:
        print(f"Error in calculate_L: {e}")
        return 0

def calculate_V(P, P0, L, t):
    """
    Calculate the value of an LP position.
    
    Parameters:
    - P: Current price
    - P0: Reference price (when LP was created/rebalanced)
    - L: Liquidity constant
    - t: Number of ticks (position width)
    
    Returns:
    - V: Current value of the LP position
    """
    try:
        # Apply the mathematical formula for position value
        return L * (2*np.sqrt(P) - (P + P0) / (np.sqrt(P0) * (1.0001 ** t)))
    except Exception as e:
        print(f"Error in calculate_V: {e}")
        return 0

def calculate_impermanent_loss(price_ratio):
    """
    Calculate impermanent loss compared to holding.
    
    Parameters:
    - price_ratio: Current price divided by initial price
    
    Returns:
    - Impermanent loss as a decimal (negative number)
    """
    # Standard formula for impermanent loss calculation
    if price_ratio <= 0:
        return 0  # Avoid division by zero or negative values
        
    sqrt_price_ratio = np.sqrt(price_ratio)
    return 2 * sqrt_price_ratio / (1 + price_ratio) - 1  # This is always <= 0

def rebalance_lp_position(price, lp_position, rebalance_cost=0):
    """
    Rebalance an LP position when price moves outside the desired range.
    
    Parameters:
    - price: Current asset price
    - lp_position: Dictionary containing the LP position data
    - rebalance_cost: Fixed cost in dollars for each rebalance
    
    Returns:
    - Updated LP position
    """
    # Check if rebalance is needed
    if (price < lp_position['p_lower_rebalance'] or 
        price > lp_position['p_upper_rebalance']):
        
        # Subtract rebalance cost
        if rebalance_cost > 0:
            lp_position['value'] -= rebalance_cost
            lp_position['rebalance_count'] += 1
        
        # Apply fee cost for rebalancing (slippage/swap fees)
        lp_position['value'] = lp_position['value'] * (1 - (0.5 - lp_position['imbalance']) * lp_position['fee_rate'])
        
        # Update position boundaries
        lp_position['p_0'] = price
        lp_position['p_lower'] = lp_position['p_0'] * (1.0001 ** (-lp_position['num_ticks']))
        lp_position['p_upper'] = lp_position['p_0'] * (1.0001 ** (lp_position['num_ticks']))
        lp_position['p_lower_rebalance'] = lp_position['p_lower'] * (1.0001 ** (lp_position['num_ticks'] * lp_position['imbalance'] * 2))
        lp_position['p_upper_rebalance'] = lp_position['p_upper'] * (1.0001 ** (-lp_position['num_ticks'] * lp_position['imbalance'] * 2))
        
        # Recalculate liquidity
        lp_position['L'] = calculate_L(lp_position['value'], lp_position['p_0'], lp_position['num_ticks'])
        lp_position['rebalanced'] = True
    else:
        lp_position['rebalanced'] = False
        
    return lp_position

def simulate_lp_value(price_path, V, t, imbalance, fee_rate, rebalance_cost=0, fee_apy=0):
    """
    Simulate an LP position over time with a given price path.
    
    Parameters:
    - price_path: Dictionary of prices over time
    - V: Initial value of the LP position
    - t: Number of ticks (position width)
    - imbalance: Threshold to trigger rebalancing
    - fee_rate: Trading fee percentage
    - rebalance_cost: Cost in dollars for each rebalance
    - fee_apy: Estimated annual fee earnings
    
    Returns:
    - DataFrame containing simulation results
    """
    # Initialize LP position
    lp_position = {
        'value': V,
        'price': price_path[0],
        'p_0': price_path[0],
        'num_ticks': t,
        'p_lower': price_path[0] * (1.0001 ** (-t)),
        'p_upper': price_path[0] * (1.0001 ** (t)),
        'imbalance': imbalance,
        'fee_rate': fee_rate,
        'rebalance_count': 0,
        'rebalanced': False,
        'fees_earned': 0,
        'time_step': 0,
        'in_range': True
    }
    
    # Calculate initial liquidity
    lp_position['L'] = calculate_L(lp_position['value'], lp_position['p_0'], lp_position['num_ticks'])
    lp_position['p_lower_rebalance'] = lp_position['p_lower'] * (1.0001 ** (t * imbalance * 2))
    lp_position['p_upper_rebalance'] = lp_position['p_upper'] * (1.0001 ** (-t * imbalance * 2))
    
    # Create DataFrame to store results
    simulation = pd.DataFrame([lp_position])
    
    # Initialize buy and hold value for comparison
    hold_value = V
    lp_position['hold_value'] = hold_value
    lp_position['impermanent_loss'] = 0
    
    # Track price movements for fee estimation
    price_changes = []
    
    # Simulate over time
    for time_step in range(1, len(price_path)):
        # Update time step
        lp_position['time_step'] = time_step
        
        # Update price
        current_price = price_path[time_step]
        prev_price = lp_position['price']
        lp_position['price'] = current_price
        
        # Calculate price movement for fee estimation
        price_change_pct = abs(current_price / prev_price - 1)
        price_changes.append(price_change_pct)
        
        # Check if price is in range
        in_range = (lp_position['p_lower'] <= current_price <= lp_position['p_upper'])
        lp_position['in_range'] = in_range
        
        # When in range, use concentrated liquidity formula
        if in_range:
            try:
                # Use the proper formula for in-range LP positions
                # This is a simplified version of the actual Uniswap V3 formula
                sqrt_current = np.sqrt(current_price)
                sqrt_p0 = np.sqrt(lp_position['p_0'])
                sqrt_lower = np.sqrt(lp_position['p_lower'])
                sqrt_upper = np.sqrt(lp_position['p_upper'])
                
                # Calculate token amounts based on current price
                if sqrt_current <= sqrt_lower:
                    # All in token 0 (base token)
                    token0 = lp_position['L'] * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper)
                    token1 = 0
                elif sqrt_current >= sqrt_upper:
                    # All in token 1 (quote token)
                    token0 = 0
                    token1 = lp_position['L'] * (sqrt_upper - sqrt_lower)
                else:
                    # Mixed holdings
                    token0 = lp_position['L'] * (sqrt_upper - sqrt_current) / (sqrt_current * sqrt_upper)
                    token1 = lp_position['L'] * (sqrt_current - sqrt_lower)
                
                # Calculate position value
                lp_value_before_fees = token0 * current_price + token1
                
                # Apply a realistic impermanent loss
                price_ratio = current_price / lp_position['p_0']
                sqrt_price_ratio = np.sqrt(price_ratio)
                true_il = 2 * sqrt_price_ratio / (1 + price_ratio) - 1
                
                # Calculate what the value would be without impermanent loss
                hold_equivalent = V * (current_price + lp_position['p_0']) / (2 * lp_position['p_0'])
                
                # Apply impermanent loss to value
                lp_value_before_fees = hold_equivalent * (1 + true_il)
            except Exception as e:
                # If there's an error in the calculation, fall back to simpler method
                lp_value_before_fees = calculate_V(current_price, 
                                                lp_position['p_0'], 
                                                lp_position['L'], 
                                                lp_position['num_ticks'])
        else:
            # Out of range - position becomes 100% one token
            if current_price > lp_position['p_upper']:  # All in quote token (e.g., USDC)
                # Constant value since all in stable token
                sqrt_price_ratio = np.sqrt(lp_position['p_upper'] / lp_position['p_0'])
                lp_value_before_fees = V * 2 * sqrt_price_ratio / (1 + (lp_position['p_upper'] / lp_position['p_0']))
            else:  # All in base token (e.g., ETH)
                # Value changes with price since all in non-stable token
                sqrt_price_ratio = np.sqrt(lp_position['p_0'] / lp_position['p_lower'])
                relative_price = current_price / lp_position['p_0']
                lp_value_before_fees = V * 2 * sqrt_price_ratio / (1 + (lp_position['p_0'] / lp_position['p_lower'])) * relative_price
        
        # Add earned fees based on price movement and range
        if in_range:
            # Fee earning is proportional to price movement (trading volume proxy)
            # and inversely proportional to range width (narrower = more fees per movement)
            width_factor = 10 / (lp_position['num_ticks'] + 10)  # Soft cap on fee multiplier
            
            # Calculate fee based on:
            # 1. Initial value
            # 2. Fee rate
            # 3. Price movement (proxy for trading volume)
            # 4. Width factor (narrower ranges earn more fees per unit of movement)
            # 5. Volume multiplier (typical ratio of volume to price movement)
            volume_multiplier = 5  # Reduced for realism
            
            fee_earned = (
                V * 
                fee_rate * 
                price_change_pct * 
                width_factor * 
                volume_multiplier
            )
            
            # Cap the fee at a reasonable maximum - no more than 0.05% of value per step
            fee_earned = min(fee_earned, V * 0.0005)
                
            lp_position['fees_earned'] += fee_earned
            lp_position['value'] = lp_value_before_fees + lp_position['fees_earned']
        else:
            lp_position['value'] = lp_value_before_fees
        
        # Update buy and hold value for comparison
        hold_value = V * (current_price / price_path[0])
        lp_position['hold_value'] = hold_value
        
        # Calculate impermanent loss
        price_ratio = current_price / price_path[0]
        sqrt_price_ratio = np.sqrt(price_ratio)
        lp_position['impermanent_loss'] = (2 * sqrt_price_ratio / (1 + price_ratio) - 1) * 100  # as percentage
        
        # Check for rebalance
        lp_position = rebalance_lp_position(current_price, lp_position, rebalance_cost)
        
        # Add to results
        new_row = pd.DataFrame([lp_position.copy()])
        simulation = pd.concat([simulation, new_row], ignore_index=True)
    
    return simulation

def simulate_multiple_lp_strategies(price_path_array, LP_initial_value, tick_spacing, 
                                   num_tick_spacing, imbalance, fee_rate, rebalance_cost, 
                                   fee_apy):
    """
    Simulate multiple LP strategies with different position widths.
    
    Parameters:
    - price_path_array: Array of prices over time
    - LP_initial_value: Initial investment amount
    - tick_spacing: Base tick spacing
    - num_tick_spacing: Number of different strategies to test
    - imbalance: Rebalance threshold
    - fee_rate: Trading fee percentage
    - rebalance_cost: Cost in dollars for each rebalance
    - fee_apy: Estimated annual fee earnings
    
    Returns:
    - Array of results
    """
    price_path_dict = {i: price_path_array[i] for i in range(len(price_path_array))}
    results = []
    
    for spacing in range(1, num_tick_spacing + 1):
        # Simulate with current strategy
        sim_result = simulate_lp_value(
            price_path_dict, 
            LP_initial_value, 
            spacing * tick_spacing, 
            imbalance, 
            fee_rate,
            rebalance_cost,
            fee_apy
        )
        
        # Extract final results
        final_row = sim_result.iloc[-1]
        
        # Add strategy details
        strategy_result = {
            'tick_width': spacing * tick_spacing,
            'price_range_percent': round((1.0001 ** (spacing * tick_spacing) - 1) * 100, 2),
            'final_value': final_row['value'],
            'initial_value': LP_initial_value,
            'return_percent': (final_row['value'] / LP_initial_value - 1) * 100,
            'hold_value': final_row['hold_value'],
            'hold_return_percent': (final_row['hold_value'] / LP_initial_value - 1) * 100,
            'relative_performance': (final_row['value'] / final_row['hold_value'] - 1) * 100,
            'rebalance_count': final_row['rebalance_count'],
            'rebalance_cost_total': final_row['rebalance_count'] * rebalance_cost,
            'fees_earned': final_row['fees_earned'],
            'impermanent_loss': final_row['impermanent_loss'],
            'price_path': sim_result['price'].tolist(),
            'value_path': sim_result['value'].tolist(),
            'hold_path': sim_result['hold_value'].tolist(),
            'time_steps': sim_result['time_step'].tolist()
        }
        
        results.append(strategy_result)
    
    return results

def simulate_price_path_worker(i, T, N, dt, mu, sigma, LP_initial_value, tick_spacing, 
                             num_tick_spacing, imbalance, fee_rate, rebalance_cost, 
                             projected_fees, starting_price):
    """Worker function for parallel simulation processing"""
    # Generate price path
    _, price_path_array = return_price_path(T, N, dt, mu, sigma, starting_price, seed=i)
    
    # Run simulation for this price path
    path_results = simulate_multiple_lp_strategies(
        price_path_array, 
        LP_initial_value, 
        tick_spacing, 
        num_tick_spacing, 
        imbalance, 
        fee_rate,
        rebalance_cost,
        projected_fees
    )
    
    # Add simulation ID to results
    for result in path_results:
        result['simulation_id'] = i
    
    return path_results

def run_simulations(T, N, dt, mu, sigma, LP_initial_value, tick_spacing, num_tick_spacing, 
                   imbalance, fee_rate, num_price_path, rebalance_cost, projected_fees, 
                   starting_price, enable_multiprocessing=True):
    """
    Run multiple price path simulations with multiple strategies.
    
    Uses multiprocessing for faster results when enabled.
    """
    all_results = []
    
    if enable_multiprocessing and num_price_path > 1:
        # Set up worker function with fixed parameters
        worker_func = partial(
            simulate_price_path_worker,
            T=T, N=N, dt=dt, mu=mu, sigma=sigma,
            LP_initial_value=LP_initial_value, tick_spacing=tick_spacing,
            num_tick_spacing=num_tick_spacing, imbalance=imbalance,
            fee_rate=fee_rate, rebalance_cost=rebalance_cost,
            projected_fees=projected_fees, starting_price=starting_price
        )
        
        # Calculate number of CPU cores to use
        num_cores = min(cpu_count(), num_price_path, 4)  # Limit to 4 cores max
        
        # Run simulations in parallel
        with Pool(num_cores) as pool:
            results = pool.map(worker_func, range(num_price_path))
            for path_results in results:
                all_results.extend(path_results)
    else:
        # Run simulations sequentially
        progress_bar = st.progress(0)
        
        for i in range(num_price_path):
            # Update progress
            progress = (i + 1) / num_price_path
            progress_bar.progress(progress, f"Running simulation {i+1}/{num_price_path}")
            
            # Generate price path
            _, price_path_array = return_price_path(T, N, dt, mu, sigma, starting_price, seed=i)
            
            # Run simulation for this price path
            path_results = simulate_multiple_lp_strategies(
                price_path_array, 
                LP_initial_value, 
                tick_spacing, 
                num_tick_spacing, 
                imbalance, 
                fee_rate,
                rebalance_cost,
                projected_fees
            )
            
            # Add simulation ID to results
            for result in path_results:
                result['simulation_id'] = i
                
            all_results.extend(path_results)
        
        # Clear progress bar when done
        progress_bar.empty()
    
    # Convert results to DataFrame
    return pd.DataFrame(all_results)

# Visualization functions
def plot_price_path(price_path, title="Asset Price Over Time"):
    """Plot the price path as a line chart"""
    fig = px.line(
        x=list(range(len(price_path))), 
        y=price_path,
        labels={"x": "Time Step", "y": "Price ($)"},
        title=title
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x",
        height=400
    )
    return fig

def plot_lp_performance(results_df, metric="return_percent", title=None):
    """Plot the performance of different LP strategies"""
    if title is None:
        title = f"LP Strategy Performance: {metric.replace('_', ' ').title()}"
    
    # Group by tick width and calculate average of the metric
    grouped = results_df.groupby('tick_width')[metric].mean().reset_index()
    
    fig = px.bar(
        grouped,
        x="tick_width",
        y=metric,
        labels={"tick_width": "Position Width (ticks)", metric: metric.replace('_', ' ').title()},
        title=title
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x",
        height=400
    )
    return fig

def plot_strategy_comparison(results_df):
    """Compare different strategies with a scatter plot"""
    # Group by tick width
    grouped = results_df.groupby('tick_width').agg({
        'return_percent': 'mean',
        'rebalance_count': 'mean',
        'impermanent_loss': 'mean',
        'fees_earned': 'mean',
        'price_range_percent': 'first'  # All values should be the same for a given tick width
    }).reset_index()
    
    # Create scatter plot
    fig = px.scatter(
        grouped,
        x="impermanent_loss",
        y="return_percent",
        size="fees_earned",
        color="rebalance_count",
        hover_name="tick_width",
        text="tick_width",
        labels={
            "impermanent_loss": "Impermanent Loss (%)",
            "return_percent": "Return (%)",
            "rebalance_count": "Avg. Rebalance Count",
            "fees_earned": "Fees Earned ($)",
            "tick_width": "Position Width (ticks)"
        },
        title="Strategy Comparison: Return vs. Impermanent Loss"
    )
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    return fig

def create_lp_animation(simulation_result, price_path, hold_values):
    """Create an animation of LP position over time"""
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(price_path))),
            y=price_path,
            mode="lines",
            name="Asset Price",
            line=dict(color="blue")
        ),
        secondary_y=True
    )
    
    # Add LP position value line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(simulation_result['value'].values))),
            y=simulation_result['value'].values,
            mode="lines",
            name="LP Position Value",
            line=dict(color="green")
        ),
        secondary_y=False
    )
    
    # Add hold strategy line
    fig.add_trace(
        go.Scatter(
            x=list(range(len(hold_values))),
            y=hold_values,
            mode="lines",
            name="Hold Strategy",
            line=dict(color="red", dash="dash")
        ),
        secondary_y=False
    )
    
    # Add position boundaries
    lower_boundaries = []
    upper_boundaries = []
    
    for _, row in simulation_result.iterrows():
        lower_boundaries.append(row['p_lower'])
        upper_boundaries.append(row['p_upper'])
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(lower_boundaries))),
            y=lower_boundaries,
            mode="lines",
            name="Lower Boundary",
            line=dict(color="gray", dash="dot")
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(upper_boundaries))),
            y=upper_boundaries,
            mode="lines",
            name="Upper Boundary",
            line=dict(color="gray", dash="dot"),
            fill='tonexty'  # Fill between upper and lower boundaries
        ),
        secondary_y=True
    )
    
    # Customize layout
    fig.update_layout(
        title="LP Position Animation",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
            }, {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
            }],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }]
    )
    
    # Set y-axis titles
    fig.update_yaxes(title_text="Position Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Asset Price ($)", secondary_y=True)
    
    return fig

def create_strategy_dashboard(results_df):
    """Create a full strategy dashboard with multiple visualizations"""
    # Group by tick width
    grouped = results_df.groupby('tick_width').agg({
        'return_percent': ['mean', 'std'],
        'hold_return_percent': ['mean', 'std'],
        'relative_performance': ['mean', 'std'],
        'rebalance_count': ['mean', 'sum'],
        'rebalance_cost_total': ['mean', 'sum'],
        'fees_earned': ['mean', 'sum'],
        'impermanent_loss': ['mean', 'std'],
        'price_range_percent': 'first'  # All should be the same for a given tick width
    }).reset_index()
    
    # Flatten the multi-index columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename columns for better readability
    grouped = grouped.rename(columns={
        'tick_width_': 'tick_width',
        'price_range_percent_first': 'price_range_percent'
    })
    
    # Create detail tables
    table_data = grouped[[
        'tick_width', 'price_range_percent', 
        'return_percent_mean', 'hold_return_percent_mean', 'relative_performance_mean',
        'rebalance_count_mean', 'fees_earned_mean', 'impermanent_loss_mean'
    ]].copy()
    
    # Round numeric columns
    for col in table_data.columns:
        if col != 'tick_width':
            table_data[col] = table_data[col].round(2)
    
    # Rename columns for the table
    table_data = table_data.rename(columns={
        'tick_width': 'Position Width (Ticks)',
        'price_range_percent': 'Price Range (%)',
        'return_percent_mean': 'Return (%)',
        'hold_return_percent_mean': 'Hold Return (%)',
        'relative_performance_mean': 'Relative Performance (%)',
        'rebalance_count_mean': 'Avg. Rebalances',
        'fees_earned_mean': 'Avg. Fees Earned ($)',
        'impermanent_loss_mean': 'Impermanent Loss (%)'
    })
    
    # Create a figure with multiple subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Return Comparison by Position Width", 
            "Fees vs. Impermanent Loss",
            "Rebalance Count by Position Width",
            "Risk/Return Profile"
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Return comparison
    fig.add_trace(
        go.Bar(
            x=grouped['tick_width'],
            y=grouped['return_percent_mean'],
            name="LP Return",
            marker_color='green',
            error_y=dict(
                type='data',
                array=grouped['return_percent_std'],
                visible=True
            )
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=grouped['tick_width'],
            y=grouped['hold_return_percent_mean'],
            name="Hold Return",
            marker_color='blue',
            error_y=dict(
                type='data',
                array=grouped['hold_return_percent_std'],
                visible=True
            )
        ),
        row=1, col=1
    )
    
    # 2. Fees vs Impermanent Loss
    fig.add_trace(
        go.Scatter(
            x=grouped['tick_width'],
            y=grouped['fees_earned_mean'],
            mode='lines+markers',
            name="Fees Earned",
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=grouped['tick_width'],
            y=grouped['impermanent_loss_mean'],
            mode='lines+markers',
            name="Impermanent Loss",
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # 3. Rebalance count
    fig.add_trace(
        go.Bar(
            x=grouped['tick_width'],
            y=grouped['rebalance_count_mean'],
            name="Avg. Rebalances",
            marker_color='orange'
        ),
        row=2, col=1
    )
    
    # 4. Risk/Return profile (scatter plot)
    fig.add_trace(
        go.Scatter(
            x=grouped['impermanent_loss_mean'],
            y=grouped['return_percent_mean'],
            mode='markers+text',
            text=grouped['tick_width'],
            textposition="top center",
            marker=dict(
                size=grouped['fees_earned_mean']/grouped['fees_earned_mean'].max()*20 + 5,
                color=grouped['price_range_percent'],
                colorscale='Viridis',
                colorbar=dict(title="Price Range (%)"),
                showscale=True
            ),
            name="Strategies"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_white",
        title="LP Strategy Analysis Dashboard",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Position Width (Ticks)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Position Width (Ticks)", row=1, col=2)
    fig.update_yaxes(title_text="Value ($) / Loss (%)", row=1, col=2)
    
    fig.update_xaxes(title_text="Position Width (Ticks)", row=2, col=1)
    fig.update_yaxes(title_text="Rebalance Count", row=2, col=1)
    
    fig.update_xaxes(title_text="Impermanent Loss (%)", row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    
    return fig, table_data

def create_live_simulation():
    """Create a live simulation with animated visualization of LP position"""
    # Parameters
    st.subheader("🎬 Live LP Position Simulation")
    st.write("Watch how an LP position behaves in real-time as prices change.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        live_tick_width = st.slider("Position Width (ticks)", 10, 200, 50, 10)
    with col2:
        live_volatility = st.slider("Market Volatility", 0.1, 2.0, 1.0, 0.1)
    with col3:
        simulation_speed = st.slider("Simulation Speed", 1, 10, 3, 1)
    
    # Initialize data container
    if 'time_step' not in st.session_state:
        st.session_state.time_step = 0
        st.session_state.price_history = [100.0]
        st.session_state.lp_value_history = [LP_initial_value]
        st.session_state.hold_value_history = [LP_initial_value]
        st.session_state.lower_boundary = [100.0 * (1.0001 ** (-live_tick_width))]
        st.session_state.upper_boundary = [100.0 * (1.0001 ** live_tick_width)]
        st.session_state.profit_history = [0.0]
        st.session_state.impermanent_loss_history = [0.0]
        st.session_state.fee_accumulation = 0.0
        
    # Define fee tier rates
    fee_tier_rates = {
        "0.01% (Very Low)": 0.0001,
        "0.05% (Low)": 0.0005,
        "0.3% (Medium)": 0.003,
        "1% (High)": 0.01
    }
        
    # Create containers for charts
    chart_container = st.container()
    metrics_container = st.container()
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Simulation", type="primary")
    with col2:
        stop_button = st.button("⏹️ Stop Simulation")
        
    if start_button:
        st.session_state.simulation_running = True
        
    if stop_button:
        st.session_state.simulation_running = False
        
    # Initialize simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
        
    # Create placeholder charts
    with chart_container:
        price_chart_placeholder = st.empty()
        value_chart_placeholder = st.empty()
        
    with metrics_container:
        metric_cols = st.columns(4)
        
    # Run simulation steps
    if st.session_state.simulation_running:
        # Update price with random walk
        current_price = st.session_state.price_history[-1]
        random_change = np.random.normal(0, live_volatility/10)
        new_price = current_price * np.exp(random_change)
        
        # Update LP position
        p_0 = st.session_state.price_history[0]
        current_step = st.session_state.time_step
        
        # Calculate LP position boundaries
        lower_price = p_0 * (1.0001 ** (-live_tick_width))
        upper_price = p_0 * (1.0001 ** live_tick_width)
        
        # Check if price is in range
        in_range = lower_price <= new_price <= upper_price
        
        # Calculate LP value properly
        price_ratio = new_price / p_0
        
        # Calculate impermanent loss (correct formula)
        sqrt_ratio = np.sqrt(price_ratio)
        impermanent_loss = 2 * sqrt_ratio / (1 + price_ratio) - 1
        
        if in_range:
            # When in range, properly calculate concentrated liquidity value
            # For concentrated liquidity in range, we need to account for the actual formula
            # This is a simplified version of the real Uniswap V3 formula
            k = LP_initial_value * 4  # constant product k = x * y
            
            # Calculate token amounts based on price
            token_x = sqrt(k / price_ratio)
            token_y = token_x * sqrt(price_ratio)
            
            # Current value is the sum of token values
            new_lp_value = token_x + token_y * price_ratio
            
            # Add accumulated fees (based on fee tier and time)
            fee_rate = fee_tier_rates.get(st.session_state.get('fee_tier', '0.3% (Medium)'), 0.003)
            # Fees are higher when price is in range and there's volatility
            fee_accumulation = LP_initial_value * fee_rate * live_volatility * 0.1 / N
            st.session_state.fee_accumulation = st.session_state.get('fee_accumulation', 0) + fee_accumulation
            new_lp_value += st.session_state.fee_accumulation
        else:
            # When out of range - the LP position behavior changes
            if new_price > upper_price:  # All in quote token (e.g., USDC)
                sqrt_price_ratio = np.sqrt(upper_price / p_0)
                new_lp_value = LP_initial_value * 2 * sqrt_price_ratio * (new_price / upper_price)
            else:  # All in base token (e.g., ETH)
                sqrt_price_ratio = np.sqrt(p_0 / lower_price)
                new_lp_value = LP_initial_value * 2 * sqrt_price_ratio * (new_price / p_0)
            
        # Calculate hold value
        new_hold_value = LP_initial_value * (new_price / p_0)
        
        # Calculate profit/loss
        profit_vs_hold = (new_lp_value / new_hold_value - 1) * 100
        
        # Update state
        st.session_state.time_step += 1
        st.session_state.price_history.append(new_price)
        st.session_state.lp_value_history.append(new_lp_value)
        st.session_state.hold_value_history.append(new_hold_value)
        st.session_state.lower_boundary.append(lower_price)
        st.session_state.upper_boundary.append(upper_price)
        st.session_state.profit_history.append(profit_vs_hold)
        st.session_state.impermanent_loss_history.append(impermanent_loss * 100)
        
        # Create price chart
        with price_chart_placeholder:
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.price_history))),
                    y=st.session_state.price_history,
                    mode="lines",
                    name="Asset Price",
                    line=dict(color="blue", width=2)
                )
            )
            
            # Add position boundaries
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.lower_boundary))),
                    y=st.session_state.lower_boundary,
                    mode="lines",
                    name="Lower Boundary",
                    line=dict(color="red", dash="dot")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.upper_boundary))),
                    y=st.session_state.upper_boundary,
                    mode="lines",
                    name="Upper Boundary",
                    line=dict(color="red", dash="dot"),
                    fill='tonexty'  # Fill between upper and lower boundaries
                )
            )
            
            fig.update_layout(
                title="Asset Price vs LP Range",
                xaxis_title="Time Step",
                yaxis_title="Price ($)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Create value chart
        with value_chart_placeholder:
            fig = go.Figure()
            
            # Add LP value line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.lp_value_history))),
                    y=st.session_state.lp_value_history,
                    mode="lines",
                    name="LP Position",
                    line=dict(color="green", width=2)
                )
            )
            
            # Add hold value line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(st.session_state.hold_value_history))),
                    y=st.session_state.hold_value_history,
                    mode="lines",
                    name="Hold Strategy",
                    line=dict(color="orange", width=2)
                )
            )
            
            fig.update_layout(
                title="LP Position vs Hold Strategy Value",
                xaxis_title="Time Step",
                yaxis_title="Value ($)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Update metrics
        with metric_cols[0]:
            st.metric(
                "Current Price", 
                f"${new_price:.2f}", 
                f"{(new_price/current_price - 1)*100:.2f}%"
            )
            
        with metric_cols[1]:
            st.metric(
                "LP Position Value", 
                f"${new_lp_value:.2f}", 
                f"{(new_lp_value/LP_initial_value - 1)*100:.2f}%"
            )
            
        with metric_cols[2]:
            st.metric(
                "Hold Strategy Value", 
                f"${new_hold_value:.2f}", 
                f"{(new_hold_value/LP_initial_value - 1)*100:.2f}%"
            )
            
        with metric_cols[3]:
            st.metric(
                "Impermanent Loss", 
                f"{impermanent_loss*100:.2f}%",
                f"{profit_vs_hold:.2f}% vs Hold"
            )
            
        # Add a small delay based on simulation speed
        time.sleep(1 / simulation_speed)
        
        # Rerun to update
        st.rerun()
    else:
        # Display static charts when not running
        if len(st.session_state.price_history) > 1:
            # Create price chart
            with price_chart_placeholder:
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.price_history))),
                        y=st.session_state.price_history,
                        mode="lines",
                        name="Asset Price",
                        line=dict(color="blue", width=2)
                    )
                )
                
                # Add position boundaries
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.lower_boundary))),
                        y=st.session_state.lower_boundary,
                        mode="lines",
                        name="Lower Boundary",
                        line=dict(color="red", dash="dot")
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.upper_boundary))),
                        y=st.session_state.upper_boundary,
                        mode="lines",
                        name="Upper Boundary",
                        line=dict(color="red", dash="dot"),
                        fill='tonexty'  # Fill between upper and lower boundaries
                    )
                )
                
                fig.update_layout(
                    title="Asset Price vs LP Range",
                    xaxis_title="Time Step",
                    yaxis_title="Price ($)",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Create value chart
            with value_chart_placeholder:
                fig = go.Figure()
                
                # Add LP value line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.lp_value_history))),
                        y=st.session_state.lp_value_history,
                        mode="lines",
                        name="LP Position",
                        line=dict(color="green", width=2)
                    )
                )
                
                # Add hold value line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(st.session_state.hold_value_history))),
                        y=st.session_state.hold_value_history,
                        mode="lines",
                        name="Hold Strategy",
                        line=dict(color="orange", width=2)
                    )
                )
                
                fig.update_layout(
                    title="LP Position vs Hold Strategy Value",
                    xaxis_title="Time Step",
                    yaxis_title="Value ($)",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Update metrics
            with metric_cols[0]:
                st.metric(
                    "Current Price", 
                    f"${st.session_state.price_history[-1]:.2f}", 
                    f"{(st.session_state.price_history[-1]/st.session_state.price_history[0] - 1)*100:.2f}%"
                )
                
            with metric_cols[1]:
                st.metric(
                    "LP Position Value", 
                    f"${st.session_state.lp_value_history[-1]:.2f}", 
                    f"{(st.session_state.lp_value_history[-1]/LP_initial_value - 1)*100:.2f}%"
                )
                
            with metric_cols[2]:
                st.metric(
                    "Hold Strategy Value", 
                    f"${st.session_state.hold_value_history[-1]:.2f}", 
                    f"{(st.session_state.hold_value_history[-1]/LP_initial_value - 1)*100:.2f}%"
                )
                
            with metric_cols[3]:
                st.metric(
                    "Impermanent Loss", 
                    f"{st.session_state.impermanent_loss_history[-1]:.2f}%",
                    f"{st.session_state.profit_history[-1]:.2f}% vs Hold"
                )
        else:
            st.info("Click 'Start Simulation' to begin the live simulation.")

# Function for range optimization
def optimize_range_width(T, N, dt, mu, sigma, LP_initial_value, tick_spacing, 
                        imbalance, fee_rate, rebalance_cost, projected_fees,
                        starting_price, num_simulations=20, progress_bar=None):
    """
    Run multiple simulations with different range widths to find the optimal width.
    
    Parameters:
    - All the standard simulation parameters
    - num_simulations: Number of price paths to simulate for each range width
    - progress_bar: Optional Streamlit progress bar to update
    
    Returns:
    - DataFrame with optimization results for different range widths
    """
    # Add more realistic range widths to test in ticks
    range_widths = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Store results
    results = []
    
    # Total progress steps
    total_steps = len(range_widths) * num_simulations
    current_step = 0
    
    # Run simulations for each range width
    for width in range_widths:
        width_results = []
        
        # Run multiple simulations for this width
        for sim_idx in range(num_simulations):
            # Generate a price path
            _, price_path_array = return_price_path(T, N, dt, mu, sigma, starting_price, seed=sim_idx)
            price_path_dict = {i: price_path_array[i] for i in range(len(price_path_array))}
            
            # Simulate this strategy
            sim_result = simulate_lp_value(
                price_path_dict, 
                LP_initial_value, 
                width * tick_spacing, 
                imbalance, 
                fee_rate,
                rebalance_cost,
                projected_fees
            )
            
            # Extract final results
            final_row = sim_result.iloc[-1]
            
            # Calculate metrics
            price_ratio = final_row['price'] / price_path_dict[0]
            sqrt_price_ratio = np.sqrt(price_ratio)
            true_il = 2 * sqrt_price_ratio / (1 + price_ratio) - 1
            
            # Calculate fees - based on price movements within range, not just time
            # First, estimate price movement within range
            price_changes = []
            in_range_count = 0
            for i in range(1, len(price_path_dict)):
                current_price = price_path_dict[i]
                prev_price = price_path_dict[i-1]
                price_change_pct = abs(current_price / prev_price - 1)
                
                # Check if price is in range
                lower_bound = price_path_dict[0] * (1.0001 ** (-width * tick_spacing))
                upper_bound = price_path_dict[0] * (1.0001 ** (width * tick_spacing))
                
                if lower_bound <= current_price <= upper_bound:
                    price_changes.append(price_change_pct)
                    in_range_count += 1
            
            # Calculate fee estimate based on price movements and time in range
            total_price_movement = sum(price_changes) if price_changes else 0
            time_in_range_pct = in_range_count / len(price_path_dict) if len(price_path_dict) > 0 else 0
            
            # Fee calculation adjustments:
            # 1. Fee tier rate
            # 2. Price movements (more movement = more fees)
            # 3. Time in range (only earn fees when in range)
            # 4. Range width factor (narrower ranges earn more fees when in range)
            # 5. Initial value
            # 6. Scale by simulation time period
            
            # Calculate width factor - narrower ranges earn proportionally more fees
            # But with a reasonable limit to avoid unrealistic fee amounts
            width_factor_base = 10 / (width + 10)  # Soft cap on fee multiplier
            
            # Base fee calculation
            base_fee_estimate = (
                LP_initial_value * 
                fee_rate * 
                total_price_movement * 
                5 *  # Volume multiplier - reduced to be more realistic
                time_in_range_pct * 
                width_factor_base
            )
            
            # Scale to realistic numbers - annual fee yield shouldn't exceed reasonable APR
            # For a high-volume market, 30-50% APR might be possible in narrow ranges
            time_scale = T * 365  # Scale to annual equivalent
            max_annual_yield = min(0.5, fee_rate * 150)  # Cap at 50% APR or fee_rate * 150
            max_fee = LP_initial_value * max_annual_yield * T
            
            # Apply reasonable caps
            fee_estimate = min(base_fee_estimate, max_fee)
            
            # Apply simulated noise to make results more realistic
            random_factor = 0.8 + (0.4 * np.random.random())  # Random factor between 0.8 and 1.2
            fee_estimate = fee_estimate * random_factor
            
            strategy_result = {
                'range_width': width * tick_spacing,
                'price_range_percent': round((1.0001 ** (width * tick_spacing) - 1) * 100, 2),
                'final_value': final_row['value'],
                'return_percent': (final_row['value'] / LP_initial_value - 1) * 100,
                'hold_value': final_row['hold_value'],
                'hold_return_percent': (final_row['hold_value'] / LP_initial_value - 1) * 100,
                'relative_performance': (final_row['value'] / final_row['hold_value'] - 1) * 100,
                'rebalance_count': final_row['rebalance_count'],
                'fees_earned': fee_estimate,
                'impermanent_loss': true_il * 100,  # Convert to percentage
                'simulation_idx': sim_idx,
                'final_price': final_row['price'],
                'price_ratio': price_ratio,
                'time_in_range_pct': time_in_range_pct * 100  # As percentage
            }
            
            width_results.append(strategy_result)
            
            # Update progress
            current_step += 1
            if progress_bar:
                progress_bar.progress(current_step / total_steps, 
                                     f"Simulating range width {width * tick_spacing} ticks, path {sim_idx+1}/{num_simulations}")
        
        # Calculate aggregate metrics for this width
        width_df = pd.DataFrame(width_results)
        width_summary = {
            'range_width': width * tick_spacing,
            'price_range_percent': round((1.0001 ** (width * tick_spacing) - 1) * 100, 2),
            'mean_return': width_df['return_percent'].mean(),
            'median_return': width_df['return_percent'].median(),
            'return_std': width_df['return_percent'].std(),
            'mean_relative_performance': width_df['relative_performance'].mean(),
            'median_relative_performance': width_df['relative_performance'].median(),
            'mean_rebalance_count': width_df['rebalance_count'].mean(),
            'mean_fees_earned': width_df['fees_earned'].mean(),
            'mean_impermanent_loss': width_df['impermanent_loss'].mean(),
            'profit_probability': (width_df['return_percent'] > 0).mean() * 100,
            'outperform_hold_probability': (width_df['relative_performance'] > 0).mean() * 100,
            'sharpe_ratio': width_df['return_percent'].mean() / (width_df['return_percent'].std() + 0.001),
            'num_simulations': num_simulations
        }
        
        results.append(width_summary)
    
    return pd.DataFrame(results)

def create_range_optimization():
    """Create a range optimization section in the app"""
    st.header("🎯 Range Width Optimization")
    st.write("""
    This tool helps you find the optimal range width for your liquidity position based on your market parameters.
    We'll run multiple simulations with different range widths and analyze which performs best.
    """)
    
    # Use parameters from sidebar but allow adjustments
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        volatility = st.slider("Market Volatility", 0.1, 2.0, sigma, 0.1,
                             help="Higher values mean more price movement")
        
        drift = st.slider("Market Drift (%)", -20.0, 20.0, mu*100, 1.0,
                        help="Expected return trend")
        drift = drift / 100  # Convert to decimal
        
        sim_fee_tier = st.selectbox(
            "Fee Tier",
            ["0.01% (Very Low)", "0.05% (Low)", "0.3% (Medium)", "1% (High)"],
            index=2,
            help="Trading fee percentage"
        )
        sim_fee_rate = fee_map[sim_fee_tier]
    
    with col2:
        num_simulations = st.slider("Number of Simulations", 10, 100, 20, 10,
                                  help="More = more reliable but slower")
        
        time_horizon = st.selectbox(
            "Time Horizon",
            ["1 Day", "1 Week", "1 Month", "3 Months"],
            index=1,
            help="How far into the future to simulate"
        )
        time_mapping_opt = {
            "1 Day": 1.0/365,
            "1 Week": 7.0/365,
            "1 Month": 30.0/365,
            "3 Months": 90.0/365
        }
        sim_T = time_mapping_opt[time_horizon]
        
        sim_rebalance_cost = st.slider("Rebalance Cost ($)", 0.0, 20.0, rebalance_cost, 1.0,
                                    help="Gas cost per rebalance")
    
    # Run optimization button
    optimize_button = st.button("🔍 Find Optimal Range", type="primary")
    
    if optimize_button:
        # Show progress bar
        progress_bar = st.progress(0, "Preparing simulations...")
        
        # Calculate simulation steps based on time horizon
        if time_horizon == "1 Day":
            sim_N = 288  # 5 minute intervals
        elif time_horizon == "1 Week":
            sim_N = 168  # 1 hour intervals
        elif time_horizon == "1 Month":
            sim_N = 720  # 1 hour intervals
        else:  # 3 Months
            sim_N = 540  # 4 hour intervals
            
        sim_dt = sim_T / sim_N
        
        # Run optimization
        with st.spinner("Running range optimization simulations..."):
            results = optimize_range_width(
                T=sim_T,
                N=sim_N,
                dt=sim_dt,
                mu=drift,
                sigma=volatility,
                LP_initial_value=LP_initial_value,
                tick_spacing=tick_spacing,
                imbalance=imbalance,
                fee_rate=sim_fee_rate,
                rebalance_cost=sim_rebalance_cost,
                projected_fees=projected_fees_annual / 365 * (sim_T * 365),
                starting_price=starting_price,
                num_simulations=num_simulations,
                progress_bar=progress_bar
            )
            
        # Clear progress bar
        progress_bar.empty()
        
        # Show optimization results
        st.subheader("Optimization Results")
        
        # Identify best options for different metrics
        best_return = results.loc[results['mean_return'].idxmax()]
        best_sharpe = results.loc[results['sharpe_ratio'].idxmax()]
        best_vs_hold = results.loc[results['mean_relative_performance'].idxmax()]
        best_fees = results.loc[results['mean_fees_earned'].idxmax()]
        
        # Display key recommendations
        rec_cols = st.columns(4)
        
        with rec_cols[0]:
            st.metric("Best Average Return", 
                      f"{int(best_return['range_width'])} ticks",
                      f"{best_return['price_range_percent']}% range")
            st.caption(f"Avg. Return: {best_return['mean_return']:.2f}%")
            
        with rec_cols[1]:
            st.metric("Best Risk-Adjusted Return", 
                      f"{int(best_sharpe['range_width'])} ticks",
                      f"Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
            st.caption(f"Avg. Return: {best_sharpe['mean_return']:.2f}%")
            
        with rec_cols[2]:
            st.metric("Best vs Hold Strategy", 
                      f"{int(best_vs_hold['range_width'])} ticks",
                      f"{best_vs_hold['outperform_hold_probability']:.1f}% win rate")
            st.caption(f"Outperforms by {best_vs_hold['mean_relative_performance']:.2f}%")
            
        with rec_cols[3]:
            st.metric("Best Fee Earnings", 
                      f"{int(best_fees['range_width'])} ticks",
                      f"${best_fees['mean_fees_earned']:.2f}")
            st.caption(f"IL: {best_fees['mean_impermanent_loss']:.2f}%")
        
        # Create performance visualization
        st.subheader("Performance by Range Width")
        
        # Return vs range width
        fig = px.line(
            results,
            x="range_width",
            y=["mean_return", "mean_impermanent_loss", "mean_fees_earned"],
            labels={
                "range_width": "Range Width (ticks)",
                "value": "Percentage / Value",
                "variable": "Metric"
            },
            title="Key Metrics by Range Width"
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk-return profile
        fig2 = px.scatter(
            results,
            x="return_std",
            y="mean_return",
            size="price_range_percent",
            color="mean_fees_earned",
            hover_name="range_width",
            text="range_width",
            labels={
                "return_std": "Risk (Return Standard Deviation)",
                "mean_return": "Average Return (%)",
                "price_range_percent": "Price Range (%)",
                "mean_fees_earned": "Average Fees Earned ($)"
            },
            title="Risk-Return Profile by Range Width"
        )
        
        fig2.update_traces(textposition="top center")
        
        fig2.update_layout(
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Optimization Results")
        
        # Select key columns for display
        display_cols = [
            'range_width', 'price_range_percent', 'mean_return', 'return_std',
            'sharpe_ratio', 'mean_relative_performance', 'mean_fees_earned',
            'mean_impermanent_loss', 'mean_rebalance_count', 'profit_probability',
            'outperform_hold_probability'
        ]
        
        # Round numeric columns
        display_df = results[display_cols].copy()
        for col in display_df.columns:
            if col not in ['range_width']:
                display_df[col] = display_df[col].round(2)
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'range_width': 'Range Width (ticks)',
            'price_range_percent': 'Price Range (%)',
            'mean_return': 'Avg Return (%)',
            'return_std': 'Return StdDev (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'mean_relative_performance': 'vs Hold (%)',
            'mean_fees_earned': 'Avg Fees ($)',
            'mean_impermanent_loss': 'Avg IL (%)',
            'mean_rebalance_count': 'Avg Rebalances',
            'profit_probability': 'Profit Chance (%)',
            'outperform_hold_probability': 'Beat Hold Chance (%)'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Final recommendation
        st.subheader("Final Recommendation")
        
        # Determine best overall strategy
        # Weight the factors:
        # - 40% Sharpe ratio rank
        # - 30% Mean return rank
        # - 20% Outperform hold rank
        # - 10% Fees earned rank
        
        results['sharpe_rank'] = results['sharpe_ratio'].rank(ascending=False)
        results['return_rank'] = results['mean_return'].rank(ascending=False)
        results['vs_hold_rank'] = results['mean_relative_performance'].rank(ascending=False)
        results['fees_rank'] = results['mean_fees_earned'].rank(ascending=False)
        
        results['overall_score'] = (
            0.4 * results['sharpe_rank'] +
            0.3 * results['return_rank'] +
            0.2 * results['vs_hold_rank'] +
            0.1 * results['fees_rank']
        )
        
        best_overall = results.loc[results['overall_score'].idxmin()]
        
        # Create recommendation text
        if volatility <= 0.3:
            market_description = "low-volatility"
        elif volatility <= 0.8:
            market_description = "medium-volatility"
        else:
            market_description = "high-volatility"
            
        if drift >= 0.05:
            trend_description = "upward trending"
        elif drift <= -0.05:
            trend_description = "downward trending"
        else:
            trend_description = "range-bound"
            
        recommendation = f"""
        Based on your {market_description}, {trend_description} market parameters, the optimal range width is:
        
        ### {int(best_overall['range_width'])} ticks (±{best_overall['price_range_percent']:.1f}% range)
        
        **Why this range works best:**
        - Expected return: {best_overall['mean_return']:.2f}%
        - Risk-adjusted return (Sharpe): {best_overall['sharpe_ratio']:.2f}
        - Probability of profit: {best_overall['profit_probability']:.1f}%
        - Probability of beating hold: {best_overall['outperform_hold_probability']:.1f}%
        - Average fees earned: ${best_overall['mean_fees_earned']:.2f}
        - Average rebalances needed: {best_overall['mean_rebalance_count']:.1f}
        
        **Market conditions considered:**
        - Volatility: {volatility} (annual)
        - Price trend: {drift*100:.1f}% (annual)
        - Time horizon: {time_horizon}
        - Fee tier: {sim_fee_tier}
        """
        
        st.markdown(recommendation)
        
        # Add an explanation about adaptation to different market conditions
        st.info("""
        **Remember**: The optimal range width changes with market conditions!
        
        - In **higher volatility** markets: Use wider ranges to reduce rebalancing frequency
        - In **lower volatility** markets: Use narrower ranges to maximize fee earnings
        - For **upward trending** markets: Position your range slightly above the current price
        - For **downward trending** markets: Position your range slightly below the current price
        """)

# Main app logic
if not run_simulation:
    # Display introduction and explanation when not running simulation
    st.header("🔍 Understanding Concentrated Liquidity Providers")
    
    st.write("""
    ### What This Tool Does
    
    This advanced simulator helps optimize Liquidity Provider (LP) strategies for protocols like Uniswap V3,
    where you can provide liquidity in concentrated price ranges. The key features include:
    
    1. **Multiple Strategy Comparison**: Test different position widths to find the optimal balance between
       fee earnings and impermanent loss.
    
    2. **Risk Analysis**: See how different strategies perform across multiple price simulations.
    
    3. **Live Simulation**: Watch how an LP position behaves in real time as prices change.
    
    4. **Realistic Parameters**: Model actual market conditions with customizable volatility,
       fee tiers, and rebalancing costs.
    
    5. **Range Optimization**: Automatically find the most profitable range for your specific market conditions.
    """)
    
    st.info("""
    **How to Use This Tool**:
    1. Configure your simulation parameters in the sidebar
    2. Click "Run Simulation" to generate results
    3. Explore the visualization tabs to analyze different strategies
    4. Try the live simulation to see how positions behave in real-time
    5. Use the range optimizer to get a specific recommendation
    """)
    
    # Add range optimization section
    st.header("🎯 Range Optimization")
    st.write("""
    Want to find the optimal range width for your liquidity position? 
    Click below to run simulations and get a data-driven recommendation.
    """)
    
    if st.button("Open Range Optimizer", type="primary"):
        st.session_state.show_optimizer = True
        
    if st.session_state.get('show_optimizer', False):
        create_range_optimization()
    
    # Display sample visualization
    st.subheader("Sample Visualization")
    sample_price_path, _ = return_price_path(T, 100, dt, mu/10, sigma, starting_price)
    sample_fig = plot_price_path(list(sample_price_path.values()), "Sample Price Path Simulation")
    st.plotly_chart(sample_fig, use_container_width=True)
    
    # Show live simulation
    create_live_simulation()
    
else:
    # Run the full simulation
    with st.spinner("Running simulation... This may take a few moments."):
        # Run simulations
        results_df = run_simulations(
            T=T,
            N=N,
            dt=dt,
            mu=mu,
            sigma=sigma,
            LP_initial_value=LP_initial_value,
            tick_spacing=tick_spacing,
            num_tick_spacing=num_tick_spacing,
            imbalance=imbalance,
            fee_rate=fee_rate,
            num_price_path=num_price_path,
            rebalance_cost=rebalance_cost,
            projected_fees=projected_fees,
            starting_price=starting_price,
            enable_multiprocessing=enable_multiprocessing
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Dashboard", 
        "📈 Performance Analysis", 
        "🔄 Rebalancing Impact", 
        "💲 Fee Analysis",
        "🎬 Live Simulation"
    ])
    
    # Tab 1: Strategy Dashboard
    with tab1:
        st.header("📊 LP Strategy Dashboard")
        
        dashboard_fig, table_data = create_strategy_dashboard(results_df)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Show the detailed data table
        st.subheader("Detailed Strategy Comparison")
        st.dataframe(table_data, use_container_width=True)
        
        # Add download buttons for the data
        csv = table_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="lp_strategy_results.csv">Download CSV Results</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Key insights
        st.subheader("🔑 Key Insights")
        
        # Find the best performing strategy
        best_return_idx = table_data["Return (%)"].idxmax()
        best_strategy = table_data.iloc[best_return_idx]
        
        # Find the most efficient strategy (highest return / IL ratio)
        table_data["Efficiency"] = table_data["Return (%)"] / (table_data["Impermanent Loss (%)"].abs() + 0.001)
        best_efficiency_idx = table_data["Efficiency"].idxmax()
        best_efficiency = table_data.iloc[best_efficiency_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Best Return Strategy:**
            - Position Width: {best_strategy['Position Width (Ticks)']} ticks
            - Price Range: ±{best_strategy['Price Range (%)']}%
            - Expected Return: {best_strategy['Return (%)']}%
            - Impermanent Loss: {best_strategy['Impermanent Loss (%)']}%
            """)
            
        with col2:
            st.info(f"""
            **Most Efficient Strategy:**
            - Position Width: {best_efficiency['Position Width (Ticks)']} ticks
            - Price Range: ±{best_efficiency['Price Range (%)']}%
            - Return/IL Ratio: {table_data['Efficiency'].iloc[best_efficiency_idx]:.2f}
            - Rebalances Needed: {best_efficiency['Avg. Rebalances']:.1f}
            """)
    
    # Tab 2: Performance Analysis
    with tab2:
        st.header("📈 Performance Analysis")
        
        # Group by simulation ID and tick width
        perf_metrics = results_df.groupby(['simulation_id', 'tick_width']).agg({
            'return_percent': 'first',
            'hold_return_percent': 'first',
            'relative_performance': 'first',
            'price_range_percent': 'first'
        }).reset_index()
        
        # Create distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Return distribution by strategy
            fig = px.box(
                perf_metrics, 
                x="tick_width", 
                y="return_percent",
                color="tick_width",
                labels={
                    "tick_width": "Position Width (ticks)",
                    "return_percent": "Return (%)"
                },
                title="Return Distribution by Strategy"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Relative performance vs hold
            fig = px.box(
                perf_metrics, 
                x="tick_width", 
                y="relative_performance",
                color="tick_width",
                labels={
                    "tick_width": "Position Width (ticks)",
                    "relative_performance": "Performance vs Hold (%)"
                },
                title="Relative Performance vs Hold Strategy"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show a scatter plot of individual simulation results
        st.subheader("Individual Simulation Results")
        
        fig = px.scatter(
            perf_metrics,
            x="price_range_percent",
            y="return_percent",
            color="hold_return_percent",
            size="relative_performance",
            facet_col="simulation_id",
            facet_col_wrap=5,
            labels={
                "price_range_percent": "Price Range (%)",
                "return_percent": "Return (%)",
                "hold_return_percent": "Hold Return (%)",
                "relative_performance": "Relative Performance (%)"
            },
            title="Strategy Performance Across All Simulations"
        )
        
        # Improve the facet layout
        fig.update_layout(
            height=800 if num_price_path > 10 else 500,
            template="plotly_white",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show a sample price path
        st.subheader("Sample Price Path")
        
        # Get a sample price path
        sample_sim_id = 0
        sample_results = results_df[results_df['simulation_id'] == sample_sim_id].iloc[0]
        sample_price = sample_results['price_path']
        
        # Plot the price path
        fig = plot_price_path(sample_price, "Sample Price Path from Simulation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Rebalancing Impact
    with tab3:
        st.header("🔄 Rebalancing Analysis")
        
        # Group by tick width
        rebalance_metrics = results_df.groupby('tick_width').agg({
            'rebalance_count': ['mean', 'min', 'max', 'std'],
            'rebalance_cost_total': ['mean', 'sum'],
            'return_percent': ['mean', 'std'],
            'price_range_percent': 'first'
        }).reset_index()
        
        # Flatten the multi-index columns
        rebalance_metrics.columns = ['_'.join(col).strip('_') for col in rebalance_metrics.columns.values]
        
        # Rename columns for better readability
        rebalance_metrics = rebalance_metrics.rename(columns={
            'tick_width_': 'tick_width',
            'price_range_percent_first': 'price_range_percent'
        })
        
        # Round numeric columns
        for col in rebalance_metrics.columns:
            if col not in ['tick_width']:
                rebalance_metrics[col] = rebalance_metrics[col].round(2)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Rebalance count by tick width
            fig = px.bar(
                rebalance_metrics,
                x="tick_width",
                y="rebalance_count_mean",
                error_y="rebalance_count_std",
                labels={
                    "tick_width": "Position Width (ticks)",
                    "rebalance_count_mean": "Average Rebalance Count"
                },
                title="Average Rebalances by Position Width"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Return vs rebalance count
            fig = px.scatter(
                rebalance_metrics,
                x="rebalance_count_mean",
                y="return_percent_mean",
                size="price_range_percent",
                color="rebalance_cost_total_mean",
                text="tick_width",
                labels={
                    "rebalance_count_mean": "Average Rebalance Count",
                    "return_percent_mean": "Average Return (%)",
                    "price_range_percent": "Price Range (%)",
                    "rebalance_cost_total_mean": "Avg. Rebalance Cost ($)",
                    "tick_width": "Position Width (ticks)"
                },
                title="Return vs Rebalance Count"
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Create a detailed rebalance analysis table
        st.subheader("Rebalance Cost Analysis")
        
        rebalance_table = rebalance_metrics[[
            'tick_width', 'price_range_percent', 'rebalance_count_mean', 
            'rebalance_count_min', 'rebalance_count_max', 'rebalance_count_std',
            'rebalance_cost_total_mean', 'return_percent_mean'
        ]].copy()
        
        # Rename columns for the table
        rebalance_table = rebalance_table.rename(columns={
            'tick_width': 'Position Width (Ticks)',
            'price_range_percent': 'Price Range (%)',
            'rebalance_count_mean': 'Avg Rebalances',
            'rebalance_count_min': 'Min Rebalances',
            'rebalance_count_max': 'Max Rebalances',
            'rebalance_count_std': 'Std Dev',
            'rebalance_cost_total_mean': 'Avg Cost ($)',
            'return_percent_mean': 'Return (%)'
        })
        
        # Calculate net return after costs
        rebalance_table['Net Return (%)'] = rebalance_table['Return (%)'] - (rebalance_table['Avg Cost ($)'] / LP_initial_value * 100)
        rebalance_table = rebalance_table.round(2)
        
        # Show the table
        st.dataframe(rebalance_table, use_container_width=True)
        
        # Add a rebalance frequency visualization
        st.subheader("Rebalance Frequency Analysis")
        
        # Get a sample simulation to analyze rebalance timing
        sample_sim_id = 0
        sample_tick = 10  # A mid-range strategy
        
        # Find the sample simulation in the results
        sample_idx = results_df[(results_df['simulation_id'] == sample_sim_id) & (results_df['tick_width'] == sample_tick * tick_spacing)].index
        
        if len(sample_idx) > 0:
            sample_result = results_df.iloc[sample_idx[0]]
            
            # Extract time steps and price path
            time_steps = sample_result['time_steps']
            price_path = sample_result['price_path']
            
            # Calculate price change percentage between steps
            price_changes = []
            for i in range(1, len(price_path)):
                price_changes.append((price_path[i] / price_path[i-1] - 1) * 100)
            
            # Create a DataFrame for analysis
            rebalance_df = pd.DataFrame({
                'time_step': time_steps[1:],
                'price': price_path[1:],
                'price_change_pct': price_changes
            })
            
            # Calculate rolling volatility
            window_size = min(20, len(rebalance_df) // 10)
            rebalance_df['rolling_volatility'] = rebalance_df['price_change_pct'].rolling(window=window_size).std()
            
            # Create volatility vs rebalance visualization
            fig = px.line(
                rebalance_df,
                x='time_step',
                y=['price_change_pct', 'rolling_volatility'],
                labels={
                    'time_step': 'Time Step',
                    'value': 'Percentage (%)',
                    'variable': 'Metric'
                },
                title="Price Changes and Volatility Over Time"
            )
            
            # Update the layout
            fig.update_layout(
                template="plotly_white",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about rebalancing
            st.info("""
            **Rebalancing Strategy Insights:**
            
            1. **Wider Positions (higher tick width)** require fewer rebalances but earn lower fees
            
            2. **Narrow Positions** earn more fees but face higher impermanent loss and rebalancing costs
            
            3. **Optimal Rebalance Threshold** depends on market volatility and gas costs:
               - In high volatility: Use wider ranges to reduce rebalance frequency
               - In low volatility: Use narrower ranges to maximize fee income
            
            4. **Gas Costs Impact:** When gas costs are high, rebalancing less frequently with wider positions
               is often more profitable
            """)
    
    # Tab 4: Fee Analysis
    with tab4:
        st.header("💲 Fee Analysis")
        
        # Group by tick width
        fee_metrics = results_df.groupby('tick_width').agg({
            'fees_earned': ['mean', 'sum', 'min', 'max', 'std'],
            'return_percent': ['mean', 'std'],
            'impermanent_loss': ['mean', 'std'],
            'price_range_percent': 'first'
        }).reset_index()
        
        # Flatten the multi-index columns
        fee_metrics.columns = ['_'.join(col).strip('_') for col in fee_metrics.columns.values]
        
        # Rename columns for better readability
        fee_metrics = fee_metrics.rename(columns={
            'tick_width_': 'tick_width',
            'price_range_percent_first': 'price_range_percent'
        })
        
        # Calculate fee vs IL ratio
        fee_metrics['fee_il_ratio'] = fee_metrics['fees_earned_mean'] / (abs(fee_metrics['impermanent_loss_mean']) + 0.001)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Fees earned by position width
            fig = px.bar(
                fee_metrics,
                x="tick_width",
                y="fees_earned_mean",
                error_y="fees_earned_std",
                labels={
                    "tick_width": "Position Width (ticks)",
                    "fees_earned_mean": "Average Fees Earned ($)"
                },
                title="Average Fees Earned by Position Width"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Fee vs IL ratio
            fig = px.line(
                fee_metrics,
                x="tick_width",
                y="fee_il_ratio",
                labels={
                    "tick_width": "Position Width (ticks)",
                    "fee_il_ratio": "Fee/IL Ratio (higher is better)"
                },
                title="Fee to Impermanent Loss Ratio"
            )
            fig.add_scatter(
                x=fee_metrics["tick_width"],
                y=fee_metrics["fee_il_ratio"],
                mode="markers",
                marker=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Fee impact analysis
        st.subheader("Fee Impact Analysis")
        
        # Calculate what percentage of return comes from fees
        fee_metrics['fee_contribution'] = fee_metrics['fees_earned_mean'] / LP_initial_value * 100
        fee_metrics['fee_percent_of_return'] = fee_metrics['fee_contribution'] / (fee_metrics['return_percent_mean'] + 0.001) * 100
        fee_metrics = fee_metrics.round(2)
        
        # Clean up extreme values
        fee_metrics['fee_percent_of_return'] = fee_metrics['fee_percent_of_return'].clip(0, 200)
        
        # Create visualization
        fig = px.bar(
            fee_metrics,
            x="tick_width",
            y=["fee_contribution", "return_percent_mean"],
            labels={
                "tick_width": "Position Width (ticks)",
                "value": "Percentage (%)",
                "variable": "Return Component"
            },
            title="Fee Contribution to Total Return"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a fee analysis table
        fee_table = fee_metrics[[
            'tick_width', 'price_range_percent', 'fees_earned_mean', 
            'fee_contribution', 'fee_percent_of_return', 'impermanent_loss_mean',
            'return_percent_mean', 'fee_il_ratio'
        ]].copy()
        
        # Rename columns for the table
        fee_table = fee_table.rename(columns={
            'tick_width': 'Position Width (Ticks)',
            'price_range_percent': 'Price Range (%)',
            'fees_earned_mean': 'Avg Fees Earned ($)',
            'fee_contribution': 'Fee Contribution (%)',
            'fee_percent_of_return': 'Fees % of Total Return',
            'impermanent_loss_mean': 'Impermanent Loss (%)',
            'return_percent_mean': 'Total Return (%)',
            'fee_il_ratio': 'Fee/IL Ratio'
        })
        
        # Show the table
        st.dataframe(fee_table, use_container_width=True)
        
        # Fee vs fee tier analysis
        st.subheader("Fee Tier Analysis")
        
        # Create sample data for different fee tiers
        fee_tiers = {
            "0.01% (Very Low)": 0.0001,
            "0.05% (Low)": 0.0005,
            "0.3% (Medium)": 0.003,
            "1% (High)": 0.01
        }
        
        # Create a DataFrame for analysis
        fee_tier_analysis = []
        
        for tier_name, tier_rate in fee_tiers.items():
            # Calculate estimated fee earnings based on the current fee rate
            scaling_factor = tier_rate / fee_rate
            
            tier_data = {
                'Fee Tier': tier_name,
                'Fee Rate': tier_rate * 100,  # Convert to percentage
                'Estimated Daily Volume ($)': LP_initial_value * 2,  # Placeholder for volume
                'Est. Annual APR (10 tick)': projected_fees_annual * scaling_factor * 100,
                'Est. Annual APR (50 tick)': projected_fees_annual * scaling_factor * 100 * (10/50),
                'Est. Annual APR (100 tick)': projected_fees_annual * scaling_factor * 100 * (10/100),
                'Est. Annual APR (200 tick)': projected_fees_annual * scaling_factor * 100 * (10/200)
            }
            
            fee_tier_analysis.append(tier_data)
        
        # Convert to DataFrame
        fee_tier_df = pd.DataFrame(fee_tier_analysis)
        
        # Create visualization
        fig = px.bar(
            fee_tier_df,
            x="Fee Tier",
            y=["Est. Annual APR (10 tick)", "Est. Annual APR (50 tick)", 
               "Est. Annual APR (100 tick)", "Est. Annual APR (200 tick)"],
            labels={
                "Fee Tier": "Fee Tier",
                "value": "Estimated Annual APR (%)",
                "variable": "Position Width"
            },
            title="Estimated APR by Fee Tier and Position Width"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fee optimization tips
        st.info("""
        **Fee Optimization Tips:**
        
        1. **Higher Fee Tiers** (0.3%, 1%) are better for:
           - Volatile tokens where traders are willing to pay more for liquidity
           - Narrower price ranges (smaller tick width)
           - Tokens with less liquidity and higher slippage
        
        2. **Lower Fee Tiers** (0.01%, 0.05%) are better for:
           - Stable pairs like stablecoin-to-stablecoin pools
           - Wider price ranges
           - High volume tokens where thin margins are acceptable
        
        3. **Optimal Position Width** decreases as:
           - Fee percentage increases
           - Market volatility decreases
           - Rebalancing costs decrease
        """)
        
        # Sample fee strategy comparison
        st.subheader("Optimal Fee Strategies by Market Scenario")
        
        # Define scenarios
        scenarios = [
            {
                "name": "Stablecoin Pair (USDC-USDT)",
                "volatility": "Very Low",
                "optimal_fee_tier": "0.01% or 0.05%",
                "optimal_width": "Very narrow (10-30 ticks)",
                "reasoning": "Prices barely move, so narrow ranges capture most volume with minimal IL risk"
            },
            {
                "name": "Major Pair (ETH-USDC)",
                "volatility": "Medium",
                "optimal_fee_tier": "0.05% or 0.3%",
                "optimal_width": "Medium (50-100 ticks)",
                "reasoning": "Balance between fee capture and reducing impermanent loss"
            },
            {
                "name": "Exotic Pair (Small-Cap Token)",
                "volatility": "High",
                "optimal_fee_tier": "0.3% or 1%",
                "optimal_width": "Wide (100-200+ ticks)",
                "reasoning": "High volatility requires wider ranges to reduce rebalancing; higher fees compensate for IL"
            },
            {
                "name": "New Token Launch",
                "volatility": "Very High",
                "optimal_fee_tier": "1%",
                "optimal_width": "Very wide (200+ ticks) or avoid LPing",
                "reasoning": "Extreme price movements make LPing risky; high fees required for compensation"
            }
        ]
        
        # Create DataFrame
        scenarios_df = pd.DataFrame(scenarios)
        
        # Show the table
        st.dataframe(scenarios_df, use_container_width=True)
    
    # Tab 5: Live Simulation
    with tab5:
        create_live_simulation()
        
    # Tab 6: Range Optimization
    with tab6:
        create_range_optimization()
    
    # Final recommendations
    st.header("🏆 Recommended Strategies")
    
    # Get top strategies by different metrics
    top_by_return = results_df.groupby('tick_width')['return_percent'].mean().reset_index()
    top_by_return = top_by_return.sort_values('return_percent', ascending=False).head(3)
    
    top_by_efficiency = results_df.groupby('tick_width').apply(
        lambda x: x['return_percent'].mean() / (abs(x['impermanent_loss'].mean()) + 0.001)
    ).reset_index()
    top_by_efficiency.columns = ['tick_width', 'efficiency']
    top_by_efficiency = top_by_efficiency.sort_values('efficiency', ascending=False).head(3)
    
    top_by_fees = results_df.groupby('tick_width')['fees_earned'].mean().reset_index()
    top_by_fees = top_by_fees.sort_values('fees_earned', ascending=False).head(3)
    
    # Create columns for different recommendations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Best Overall Return")
        best_return_width = top_by_return.iloc[0]['tick_width']
        best_return_value = top_by_return.iloc[0]['return_percent']
        
        # Get additional details about this strategy
        best_return_details = results_df[results_df['tick_width'] == best_return_width].iloc[0]
        
        st.metric(
            f"Strategy: {best_return_width} ticks",
            f"{best_return_value:.2f}% Return",
            f"{best_return_details['price_range_percent']:.2f}% Range"
        )
        
        st.write(f"Avg Fees Earned: ${best_return_details['fees_earned']:.2f}")
        st.write(f"Impermanent Loss: {best_return_details['impermanent_loss']:.2f}%")
        st.write(f"Rebalances Needed: {best_return_details['rebalance_count']:.1f}")
    
    with col2:
        st.subheader("Most Efficient Strategy")
        best_efficiency_width = top_by_efficiency.iloc[0]['tick_width']
        best_efficiency_value = top_by_efficiency.iloc[0]['efficiency']
        
        # Get additional details about this strategy
        best_efficiency_details = results_df[results_df['tick_width'] == best_efficiency_width].iloc[0]
        
        st.metric(
            f"Strategy: {best_efficiency_width} ticks",
            f"{best_efficiency_details['return_percent']:.2f}% Return",
            f"{best_efficiency_details['price_range_percent']:.2f}% Range"
        )
        
        st.write(f"Return/IL Ratio: {best_efficiency_value:.2f}")
        st.write(f"Impermanent Loss: {best_efficiency_details['impermanent_loss']:.2f}%")
        st.write(f"Rebalances Needed: {best_efficiency_details['rebalance_count']:.1f}")
    
    with col3:
        st.subheader("Best Fee Earner")
        best_fees_width = top_by_fees.iloc[0]['tick_width']
        best_fees_value = top_by_fees.iloc[0]['fees_earned']
        
        # Get additional details about this strategy
        best_fees_details = results_df[results_df['tick_width'] == best_fees_width].iloc[0]
        
        st.metric(
            f"Strategy: {best_fees_width} ticks",
            f"${best_fees_value:.2f} Fees",
            f"{best_fees_details['price_range_percent']:.2f}% Range"
        )
        
        st.write(f"Total Return: {best_fees_details['return_percent']:.2f}%")
        st.write(f"Impermanent Loss: {best_fees_details['impermanent_loss']:.2f}%")
        st.write(f"Rebalances Needed: {best_fees_details['rebalance_count']:.1f}")
    
    # Add a conclusion
    st.markdown("""
    ### Key Takeaways
    
    Based on the simulations with your selected parameters:
    
    1. **Trade-offs exist** between earning fees (narrower positions) and minimizing impermanent loss (wider positions)
    
    2. **Rebalancing costs** significantly impact overall profitability, especially for narrower positions
    
    3. **Optimal strategies vary** based on:
       - Market volatility
       - Fee tier selection
       - Rebalancing costs
       - Price trajectory
    
    Adjust your parameters to simulate different market scenarios and find the best strategy for your specific situation.
    """)
    
    # Add a download button for the full results
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="full_simulation_results.csv">Download Complete Simulation Data</a>'
    st.markdown(href, unsafe_allow_html=True)

# Add footer
st.markdown("""
---
### About This Tool

This advanced LP simulation dashboard was created to help optimize liquidity provider strategies
for concentrated liquidity protocols like Uniswap V3. The simulation uses geometric Brownian motion
to model price movements and realistic fee accumulation and rebalancing mechanics.

**Parameters Explained:**
- **Time Period**: Duration of the simulation (days/weeks/months)
- **Volatility**: How much prices fluctuate (higher = more movement)
- **Position Width**: Size of the LP price range (higher = wider range, lower = more concentrated)
- **Rebalance Threshold**: How far price can move before triggering a position adjustment

For more advanced usage, check the "Show Advanced Options" checkbox in the sidebar.
""")

# Add help text
with st.sidebar.expander("Help & Explanations"):
    st.markdown("""
    ### LP Position Terms
    
    - **Tick Width**: The size of your liquidity position's price range.
    - **Impermanent Loss**: Potential loss compared to holding when prices change.
    - **Rebalancing**: Adjusting your position when price moves outside your range.
    - **Geometric Brownian Motion**: Mathematical model for simulating price movements.
    
    ### Tips for Using This Tool
    
    1. Start with realistic parameters based on historical data.
    2. Run multiple simulations to account for randomness.
    3. Compare different strategies across various metrics.
    4. Test how changing parameters affects optimal position width.
    
    ### What Each Tab Shows
    
    - **Strategy Dashboard**: Overall performance comparison across strategies.
    - **Performance Analysis**: Detailed view of returns and distributions.
    - **Rebalancing Impact**: How rebalancing affects costs and returns.
    - **Fee Analysis**: In-depth look at fee earnings and optimization.
    - **Live Simulation**: Watch a position perform in real-time.
    """)
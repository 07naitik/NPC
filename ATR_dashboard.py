import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="SOL LP Width Optimizer",
    page_icon="📊",
    layout="wide"
)

# Initialize connection variable
db_connection = None

# Set up functions for database connection (reusing provided code)
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

# Function to fetch latest ATR and matching historical days
def fetch_atr_data(connection, atr_column, lookback_days):
    """
    Fetch the latest ATR value and historical days with similar ATR.
    
    Args:
        connection: Database connection
        atr_column: Column name for ATR (e.g., 'atr6', 'atr10')
        lookback_days: Number of days to look back
        
    Returns:
        latest_atr: The latest ATR value
        similar_days_data: DataFrame containing days with similar ATR and rebalance counts
    """
    if connection is None:
        st.error("Cannot fetch ATR data: No database connection")
        return None, None
    
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Use a fixed date (April 16, 2025) for current date reference
        reference_date = datetime(2025, 4, 16)
        
        # Get the latest ATR value
        cursor.execute(f"""
            SELECT time, {atr_column} 
            FROM sol_daily_price_atr_data
            WHERE time <= %s
            ORDER BY time DESC
            LIMIT 1
        """, (reference_date,))
        
        latest_data = cursor.fetchone()
        
        if not latest_data:
            st.error(f"No data found for {atr_column}.")
            cursor.close()
            return None, None
        
        latest_atr = latest_data[atr_column]
        latest_time = latest_data['time']
        
        # Calculate the start date for the lookback period
        start_time = latest_time - timedelta(days=lookback_days)
        
        # Calculate ATR range (±20%)
        atr_min = latest_atr * 0.8
        atr_max = latest_atr * 1.2
        
        # Fetch historical days with similar ATR
        cursor.execute(f"""
            SELECT time, {atr_column}, 
                   rebalance_count_0_5, rebalance_count_1, 
                   rebalance_count_2, rebalance_count_3, 
                   rebalance_count_4, rebalance_count_5,
                   rebalance_count_6
            FROM sol_daily_price_atr_data
            WHERE time >= %s AND time < %s
            AND {atr_column} BETWEEN %s AND %s
            ORDER BY time DESC
        """, (start_time, latest_time, atr_min, atr_max))
        
        similar_days = cursor.fetchall()
        cursor.close()
        
        if not similar_days:
            st.warning(f"No historical days found with ATR similar to current value ({latest_atr:.4f}).")
            return latest_atr, None
        
        # Convert to DataFrame
        similar_days_df = pd.DataFrame(similar_days)
        
        return latest_atr, similar_days_df
        
    except Exception as e:
        st.error(f"Error fetching ATR data: {str(e)}")
        if 'cursor' in locals() and cursor is not None:
            cursor.close()
        return None, None

# Function to calculate average rebalance frequencies and estimated metrics
def calculate_metrics(similar_days_df, rebalance_costs):
    """
    Calculate average rebalance frequencies and other metrics.
    
    Args:
        similar_days_df: DataFrame with historical days data
        rebalance_costs: Dictionary of rebalance costs by width
        
    Returns:
        metrics_df: DataFrame with calculated metrics
    """
    if similar_days_df is None or similar_days_df.empty:
        return None
    
    # Width values in percentage
    widths = [0.5, 1, 2, 3, 4, 5, 6]
    
    # Initialize metrics DataFrame
    metrics_data = []
    
    for width in widths:
        # Convert width to column name format
        if width == 0.5:
            col_name = 'rebalance_count_0_5'
        else:
            col_name = f'rebalance_count_{int(width)}'
        
        # Calculate average rebalance frequency
        avg_rebalance = similar_days_df[col_name].mean()
        
        # Calculate estimated losses based on rebalance costs
        estimated_loss = avg_rebalance * rebalance_costs.get(width, 0)
        
        # Add to metrics data
        metrics_data.append({
            'width': width,
            'avg_rebalance_frequency': avg_rebalance,
            'estimated_loss': estimated_loss
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    
    return metrics_df

# Function to calculate optimal LP width
def calculate_optimal_width(metrics_df, expected_yields):
    """
    Calculate the optimal LP width based on metrics and expected yields.
    
    Args:
        metrics_df: DataFrame with calculated metrics
        expected_yields: Dictionary of expected yields by width
        
    Returns:
        metrics_df: Updated DataFrame with net profit and optimal flag
        optimal_width: The optimal LP width
    """
    if metrics_df is None or metrics_df.empty:
        return None, None
    
    # Add expected yields and calculate net profit
    metrics_df['expected_yield'] = metrics_df['width'].map(expected_yields)
    metrics_df['net_profit'] = metrics_df['expected_yield'] - metrics_df['estimated_loss']
    
    # Find the optimal width (highest net profit)
    optimal_idx = metrics_df['net_profit'].idxmax()
    optimal_width = metrics_df.loc[optimal_idx, 'width']
    
    # Add optimal flag
    metrics_df['is_optimal'] = metrics_df['width'] == optimal_width
    
    return metrics_df, optimal_width

# Function to create visualization
def create_visualization(metrics_df):
    """
    Create a Plotly visualization of the metrics.
    
    Args:
        metrics_df: DataFrame with calculated metrics
        
    Returns:
        fig: Plotly figure
    """
    if metrics_df is None or metrics_df.empty:
        return None
    
    # Create figure with multiple traces
    fig = go.Figure()
    
    # Add traces for each metric
    fig.add_trace(go.Scatter(
        x=metrics_df['width'],
        y=metrics_df['avg_rebalance_frequency'],
        mode='lines+markers',
        name='Avg. Rebalance Frequency',
        line=dict(color='blue', width=2),
        hovertemplate='Width: ±%{x}%<br>Avg. Rebalances: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=metrics_df['width'],
        y=metrics_df['estimated_loss'],
        mode='lines+markers',
        name='Estimated Loss (%)',
        line=dict(color='red', width=2),
        hovertemplate='Width: ±%{x}%<br>Est. Loss: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=metrics_df['width'],
        y=metrics_df['expected_yield'],
        mode='lines+markers',
        name='Expected Yield (%)',
        line=dict(color='green', width=2),
        hovertemplate='Width: ±%{x}%<br>Exp. Yield: %{y:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=metrics_df['width'],
        y=metrics_df['net_profit'],
        mode='lines+markers',
        name='Net Profit (%)',
        line=dict(color='purple', width=3),
        hovertemplate='Width: ±%{x}%<br>Net Profit: %{y:.2f}%<extra></extra>'
    ))
    
    # Highlight optimal width
    if 'is_optimal' in metrics_df.columns:
        optimal_row = metrics_df[metrics_df['is_optimal']]
        if not optimal_row.empty:
            width_val = optimal_row['width'].values[0]
            profit_val = optimal_row['net_profit'].values[0]
            
            fig.add_trace(go.Scatter(
                x=[width_val],
                y=[profit_val],
                mode='markers',
                marker=dict(size=15, color='gold', symbol='star'),
                name='Optimal Width',
            ))
    
    # Update layout
    fig.update_layout(
        title='LP Width Optimization Metrics',
        xaxis_title='LP Width (%)',
        yaxis_title='Value',
        legend=dict(x=0.01, y=0.99, bordercolor='Black', borderwidth=1),
        hovermode='x unified',
        height=600
    )
    
    # Add grid and adjust margins
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            tickvals=metrics_df['width']
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)',
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.2)',
            zerolinewidth=1.5
        )
    )
    
    return fig

# Main Streamlit app
def main():
    # App title and description
    st.title("SOL LP Width Optimizer")
    st.markdown("""
    This dashboard helps you select the optimal liquidity provision (LP) width for SOL based on:
    - Historical Average True Range (ATR)
    - Historical rebalance frequency data
    - Expected rebalance costs and yields
    """)
    
    # Establish database connection
    connection = setup_timescale_connection()
    
    # User input section
    st.header("Input Parameters")
    
    # Create columns for user inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ATR timeframe selection
        atr_options = {
            'ATR 6': 'atr6',
            'ATR 10': 'atr10',
            'ATR 14': 'atr14',
            'ATR 21': 'atr21'
        }
        selected_atr = st.selectbox(
            "Select ATR Timeframe",
            list(atr_options.keys())
        )
        atr_column = atr_options[selected_atr]
        
    with col2:
        # Historical lookback period
        lookback_days = st.slider(
            "Historical Lookback (days)",
            min_value=30,
            max_value=1500,
            value=90,
            step=30
        )
        
        # Display reference date information
        st.info("Using April 16, 2025 as reference date")
        
    with col3:
        st.subheader("Current Status")
        # This will be populated after data is fetched
    
    # Rebalance costs input
    st.subheader("Expected Rebalance Costs (Losses + Swap Fees)")
    st.markdown("Enter the expected cost (%) for each rebalance event at different LP widths.")
    
    # Create multiple columns for cost inputs
    cost_cols = st.columns(7)
    rebalance_costs = {}
    
    widths = [0.5, 1, 2, 3, 4, 5, 6]
    for i, width in enumerate(widths):
        with cost_cols[i]:
            cost = st.number_input(
                f"±{width}% Width",
                min_value=0.0,
                max_value=10.0,
                value=float(width),  # Default to the width value itself
                step=0.1,
                format="%.2f"
            )
            rebalance_costs[width] = cost
    
    # Expected yield input
    st.subheader("Expected Yield for Next 24 Hours (%)")
    st.markdown("Enter the expected yield (%) for each LP width as shown in Orca.")
    
    # Create multiple columns for yield inputs
    yield_cols = st.columns(7)
    expected_yields = {}
    
    # Default values from the image
    default_yields = {
        0.5: 9.50,
        1.0: 5.00,
        2.0: 2.50,
        3.0: 1.60,
        4.0: 1.25,
        5.0: 1.00,
        6.0: 0.80
    }
    
    for i, width in enumerate(widths):
        with yield_cols[i]:
            yield_val = st.number_input(
                f"±{width}% Width",
                min_value=0.0,
                max_value=100.0,
                value=default_yields[width],  # Use values from the image
                step=0.1,
                format="%.2f"
            )
            expected_yields[width] = yield_val
    
    # Fetch data button
    if st.button("Calculate Optimal LP Width", type="primary"):
        # Fetch ATR data and similar historical days
        with st.spinner("Fetching ATR data and analyzing historical patterns..."):
            latest_atr, similar_days_df = fetch_atr_data(connection, atr_column, lookback_days)
            
            if latest_atr and similar_days_df is not None:
                # Calculate metrics
                metrics_df = calculate_metrics(similar_days_df, rebalance_costs)
                metrics_df, optimal_width = calculate_optimal_width(metrics_df, expected_yields)
                
                # Display current status
                with col3:
                    st.metric(
                        label=f"Latest {selected_atr} Value",
                        value=f"{latest_atr:.4f}"
                    )
                    st.metric(
                        label="Similar Historical Days",
                        value=len(similar_days_df)
                    )
                    # Show the reference date being used
                    reference_date = datetime(2025, 4, 16)
                    st.metric(
                        label="Reference Date",
                        value=reference_date.strftime("%Y-%m-%d")
                    )
                
                # Display results
                st.header("Analysis Results")
                
                # Display optimal width
                if optimal_width:
                    st.success(f"### Optimal LP Width: ±{optimal_width}%")
                    
                    # Create and display metrics table
                    st.subheader("Detailed Metrics by LP Width")
                    
                    # Format the metrics DataFrame for display
                    display_df = metrics_df.copy()
                    display_df['width'] = display_df['width'].apply(lambda x: f"±{x}%")
                    display_df['avg_rebalance_frequency'] = display_df['avg_rebalance_frequency'].round(2)
                    display_df['estimated_loss'] = display_df['estimated_loss'].round(2)
                    display_df['expected_yield'] = display_df['expected_yield'].round(2)
                    display_df['net_profit'] = display_df['net_profit'].round(2)
                    
                    # Rename columns for display
                    display_df = display_df.rename(columns={
                        'width': 'LP Width',
                        'avg_rebalance_frequency': 'Avg. Rebalance Frequency',
                        'estimated_loss': 'Estimated Loss (%)',
                        'expected_yield': 'Expected Yield (%)',
                        'net_profit': 'Net Profit (%)'
                    })
                    
                    # Drop the is_optimal column
                    display_df = display_df.drop(columns=['is_optimal'])
                    
                    # Display styled table
                    st.dataframe(display_df)
                    
                    # Create and display visualization
                    fig = create_visualization(metrics_df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not determine optimal LP width from the data.")
            else:
                st.error("Could not fetch necessary data from the database.")
    
    # Display information about the calculation logic
    with st.expander("How is the optimal LP width calculated?"):
        st.markdown("""
        ### Calculation Logic
        
        1. **ATR Matching**: We find historical days where the ATR was within ±20% of today's ATR value.
        
        2. **Rebalance Frequency**: For each width, we calculate the average number of rebalances that occurred on those similar days.
        
        3. **Loss Estimation**: We multiply the average rebalance frequency by the user-provided rebalance cost to estimate potential losses.
        
        4. **Net Profit Calculation**: We subtract the estimated losses from the expected yield for each width.
        
        5. **Optimal Width**: The width with the highest net profit is identified as the optimal LP width.
        
        This approach helps you balance the trade-off between higher yields (typically at narrower widths) and increased rebalancing costs due to price movements outside the range.
        """)

if __name__ == "__main__":
    main()

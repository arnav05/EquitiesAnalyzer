import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from swing_trading_algorithm import SwingTradingAlgorithm
import sys
from io import StringIO


st.set_page_config(page_title="Swing Trading Algorithm", layout="wide")

# Make the Go button the same height as Streamlit inputs/selects
st.markdown(
    """
    <style>
    /* Align the Go button with the selectbox input (not the label container) */
    .stButton { margin-top: 28px; }          /* offset equals label height above select */
    .stButton > button {
        height: 48px;                        /* match select input height */
        padding-top: 0; padding-bottom: 0;   /* keep exact height */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 AI-Powered Swing Trading Algorithm")

# Top controls
col1, col2, col3 = st.columns([6, 2, 1])
with col1:
    ticker = st.text_input("Ticker Symbol", placeholder="Enter ticker symbol (e.g., AAPL, TSLA, MSFT)", label_visibility="hidden", key="ticker_input")
with col2:
    timeframe = st.selectbox(
        "Time Frame:", ["1 Year", "2 Years", "5 Years", "10 Years"], index=1
    )
with col3:
    go_clicked = st.button("Go", use_container_width=True)

# Parse timeframe to years
timeframe_map = {"1 Year": 1, "2 Years": 2, "5 Years": 5, "10 Years": 10}
years = timeframe_map[timeframe]

# Run algorithm when button is clicked
if go_clicked and ticker:
    ticker = ticker.strip().upper()
    
    if not ticker:
        st.error("Please enter a valid ticker symbol")
    else:
        try:
            # Create progress indicator
            with st.spinner(f"Running swing trading algorithm for {ticker} over {years} year(s)..."):
                # Capture console output
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                # Create and run algorithm
                algo = SwingTradingAlgorithm(ticker=ticker, years=years)
                results = algo.run()
                
                # Restore stdout
                sys.stdout = old_stdout
                console_output = captured_output.getvalue()
            
            # Display success message
            st.success(f"✅ Analysis complete for {ticker}!")
            
            # Display performance metrics in columns
            st.subheader("📊 Performance Summary")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="Total Return",
                    value=f"{results['total_return']:+.2f}%",
                    delta=f"${results['final_value'] - results['initial_capital']:.2f}"
                )
            
            with metric_col2:
                st.metric(
                    label="Final Portfolio Value",
                    value=f"${results['final_value']:.2f}",
                    delta=f"From ${results['initial_capital']:.2f}"
                )
            
            with metric_col3:
                st.metric(
                    label="Total Trades",
                    value=results['total_trades'],
                    delta=f"{results['sell_trades']} completed"
                )
            
            with metric_col4:
                st.metric(
                    label="Win Rate",
                    value=f"{results['win_rate']:.1f}%"
                )
            
            # Display the interactive candlestick chart
            st.subheader("📈 Trading Signals & Portfolio Performance")
            fig = algo.plot_candlestick_with_signals()
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance
            st.subheader("🔍 Model Insights")
            col_feat1, col_feat2 = st.columns([1, 1])
            
            with col_feat1:
                st.markdown("**Feature Importance**")
                importances = algo.model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': algo.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig_importance = go.Figure(data=[
                    go.Bar(
                        x=importance_df['Importance'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker=dict(color='#2F80ED')
                    )
                ])
                fig_importance.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=300
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col_feat2:
                st.markdown("**Recent Trades**")
                if results['trades']:
                    # Show last 10 trades
                    trades_df = pd.DataFrame(results['trades'][-10:])
                    trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                    trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
                    trades_df['shares'] = trades_df['shares'].apply(lambda x: f"{x:.4f}")
                    trades_df['value'] = trades_df['value'].apply(lambda x: f"${x:.2f}")
                    
                    # Rename columns for cleaner display
                    display_df = trades_df[['date', 'action', 'price', 'shares', 'value']].rename(columns={
                        'date': 'Date',
                        'action': 'Action',
                        'price': 'Price',
                        'shares': 'Shares',
                        'value': 'Total Value'
                    })
                    
                    # Style the dataframe with colored rows
                    def highlight_action(row):
                        if row['Action'] == 'BUY':
                            return ['background-color: #d4edda; color: #155724'] * len(row)
                        elif row['Action'] == 'SELL':
                            return ['background-color: #f8d7da; color: #721c24'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = display_df.style.apply(highlight_action, axis=1)
                    st.dataframe(styled_df, use_container_width=True, hide_index=True, height=300)
                else:
                    st.info("No trades executed in this period")
            
            # Expandable section for detailed logs
            with st.expander("📋 View Detailed Logs"):
                st.code(console_output, language=None)
                
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check the ticker symbol and try again.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")

elif go_clicked and not ticker:
    st.warning("⚠️ Please enter a ticker symbol")

# Show default message when nothing has been run yet
if not go_clicked:
    st.info("👆 Enter a stock ticker symbol and select a timeframe to get started")
    
    # Show example
    st.markdown("---")
    st.markdown("""
    ### How it works:
    1. **Enter a ticker symbol** (e.g., AAPL for Apple, TSLA for Tesla)
    2. **Select a timeframe** (1-10 years of historical data)
    3. **Click 'Go'** to run the analysis
    
    The algorithm will:
    - Fetch historical stock data
    - Calculate technical indicators (RSI, MACD, Bollinger Bands, OBV, ATR)
    - Train a Random Forest model to predict Buy/Hold/Sell signals
    - Backtest the strategy with $1,000 initial capital
    - Display interactive charts and performance metrics
    """)



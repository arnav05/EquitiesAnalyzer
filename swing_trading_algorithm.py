"""
Swing Trading Algorithm using Random Forest and Technical Indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time

warnings.filterwarnings('ignore')

# Try to import yfinance exceptions if available
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = None


class SwingTradingAlgorithm:
    """
    A swing trading algorithm that uses technical indicators and Random Forest
    to predict Buy/Hold/Sell signals.
    """
    
    def __init__(self, ticker, years=2):
        """
        Initialize the trading algorithm.
        
        Args:
            ticker (str): Stock symbol (e.g., "AAPL")
            years (int): Number of years of historical data (1-10)
        """
        self.ticker = ticker
        self.years = years
        self.data = None
        self.model = None
        self.features = ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Middle', 
                        'BB_Lower', 'OBV', 'ATR']
        self.initial_capital = 1000
        
    def fetch_data(self, max_retries=3, retry_delay=5):
        """
        Fetch historical OHLCV data using yfinance with retry logic for rate limits.
        
        Args:
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Initial delay in seconds between retries (exponential backoff)
        """
        print(f"Fetching {self.years} years of data for {self.ticker}...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)
        
        # Retry logic for rate limit errors
        for attempt in range(max_retries):
            # Add delay before each attempt (longer for first attempt)
            if attempt == 0:
                time.sleep(2)
            elif attempt > 0:
                # Already handled in the exception/empty data cases below
                pass
            
            try:
                # Try using Ticker object first (more reliable)
                ticker_obj = yf.Ticker(self.ticker)
                
                # Download data - try both methods
                try:
                    self.data = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'), 
                                                   end=end_date.strftime('%Y-%m-%d'))
                except Exception as e1:
                    # Fallback to download method
                    try:
                        self.data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
                    except Exception as e2:
                        # If both fail, raise the first exception
                        raise e1
                
                # Check if data is empty (could be rate limit or invalid ticker)
                if self.data.empty or len(self.data) == 0:
                    # If it's not the last attempt, treat empty data as potential rate limit
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⚠️ No data returned (possible rate limit). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Verify ticker is valid (but don't fail if info is also rate limited)
                        try:
                            info = ticker_obj.info
                            if info and 'symbol' in info and info.get('symbol') != self.ticker:
                                raise ValueError(f"Invalid ticker symbol: {self.ticker}. Please check the symbol and try again.")
                        except:
                            pass  # If info fetch fails, assume it's rate limiting
                        
                        raise ValueError(
                            f"No data found for ticker {self.ticker} after {max_retries} attempts. "
                            f"This is likely due to Yahoo Finance rate limiting. Please wait 5-10 minutes and try again."
                        )
                
                # Flatten multi-level columns if present (newer yfinance versions)
                if isinstance(self.data.columns, pd.MultiIndex):
                    self.data.columns = self.data.columns.get_level_values(0)
                
                print(f"Downloaded {len(self.data)} days of data")
                return self.data
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Check if it's a rate limit error (multiple ways yfinance can report this)
                is_rate_limit = (
                    (YFRateLimitError is not None and isinstance(e, YFRateLimitError)) or
                    "Rate limited" in error_msg or 
                    "Too Many Requests" in error_msg or 
                    "YFRateLimitError" in error_msg or
                    "YFRateLimitError" in error_type or
                    "429" in error_msg or
                    "rate limit" in error_msg.lower()
                )
                
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise ValueError(
                            f"❌ Rate limit error after {max_retries} attempts. "
                            f"Yahoo Finance is temporarily limiting requests. "
                            f"Please wait 5-10 minutes and try again, or use a different ticker."
                        )
                else:
                    # Re-raise if it's a different error
                    raise ValueError(f"Error fetching data for {self.ticker}: {error_msg}")
        
        raise ValueError(f"Failed to fetch data for {self.ticker} after {max_retries} attempts")
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = self.data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        middle_band = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_obv(self):
        """Calculate On Balance Volume (OBV)."""
        obv = [0]
        close_values = self.data['Close'].values
        volume_values = self.data['Volume'].values
        
        for i in range(1, len(self.data)):
            if close_values[i] > close_values[i-1]:
                obv.append(obv[-1] + volume_values[i])
            elif close_values[i] < close_values[i-1]:
                obv.append(obv[-1] - volume_values[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=self.data.index)
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range (ATR)."""
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def compute_technical_indicators(self):
        """Compute all technical indicators."""
        print("Computing technical indicators...")
        
        # RSI
        self.data['RSI'] = self.calculate_rsi()
        
        # MACD
        self.data['MACD'], self.data['MACD_Signal'] = self.calculate_macd()
        
        # Bollinger Bands
        self.data['BB_Upper'], self.data['BB_Middle'], self.data['BB_Lower'] = \
            self.calculate_bollinger_bands()
        
        # OBV
        self.data['OBV'] = self.calculate_obv()
        
        # ATR
        self.data['ATR'] = self.calculate_atr()
        
        print("Technical indicators computed successfully")
    
    def define_labels(self, forward_days=3, threshold=0.01):
        """
        Define Buy/Hold/Sell signals based on future returns.
        
        Args:
            forward_days (int): Days to look ahead for return calculation
            threshold (float): Percentage threshold for buy/sell signals (0.01 = 1%)
        """
        print(f"Defining labels with {forward_days}-day forward return and {threshold*100}% threshold...")
        
        # Calculate future return
        self.data['Future_Close'] = self.data['Close'].shift(-forward_days)
        self.data['Future_Return'] = (self.data['Future_Close'] - self.data['Close']) / self.data['Close']
        
        # Define signals
        self.data['Signal'] = 0  # Hold by default
        self.data.loc[self.data['Future_Return'] >= threshold, 'Signal'] = 1  # Buy
        self.data.loc[self.data['Future_Return'] <= -threshold, 'Signal'] = -1  # Sell
        
        # Drop rows with NaN values
        self.data = self.data.dropna()
        
        signal_counts = self.data['Signal'].value_counts()
        print(f"Signal distribution - Buy: {signal_counts.get(1, 0)}, "
              f"Hold: {signal_counts.get(0, 0)}, Sell: {signal_counts.get(-1, 0)}")
    
    def train_model(self):
        """Train Random Forest classifier."""
        print("Training Random Forest model...")
        
        # Prepare features and target
        X = self.data[self.features]
        y = self.data['Signal']
        
        # Train model on all data
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        # Calculate accuracy
        accuracy = self.model.score(X, y)
        print(f"Model training complete. Accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def predict_signals(self):
        """Predict Buy/Hold/Sell signals using the trained model."""
        print("Predicting signals...")
        
        X = self.data[self.features]
        self.data['Predicted_Signal'] = self.model.predict(X)
        
        pred_counts = self.data['Predicted_Signal'].value_counts()
        print(f"Prediction distribution - Buy: {pred_counts.get(1, 0)}, "
              f"Hold: {pred_counts.get(0, 0)}, Sell: {pred_counts.get(-1, 0)}")
    
    def backtest(self):
        """
        Simulate trading with $1000 initial capital.
        
        Returns:
            dict: Performance metrics
        """
        print(f"\nBacktesting with ${self.initial_capital} initial capital...")
        
        cash = self.initial_capital
        shares = 0
        trades = []
        portfolio_values = []
        
        for idx, row in self.data.iterrows():
            signal = row['Predicted_Signal']
            price = row['Close']
            
            # Execute trades based on signals
            if signal == 1 and cash > 0:  # Buy signal
                shares_to_buy = cash / price
                shares += shares_to_buy
                trades.append({
                    'date': idx,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'value': cash
                })
                cash = 0
                
            elif signal == -1 and shares > 0:  # Sell signal
                cash = shares * price
                trades.append({
                    'date': idx,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares,
                    'value': cash
                })
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = cash + (shares * price)
            portfolio_values.append(portfolio_value)
        
        # Final liquidation
        final_price = self.data['Close'].iloc[-1]
        final_value = cash + (shares * final_price)
        
        # Calculate metrics
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate win rate
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        profitable_trades = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['value'] > buy_trades[i]['value']:
                profitable_trades += 1
        
        win_rate = (profitable_trades / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
        
        # Store portfolio values
        self.data['Portfolio_Value'] = portfolio_values
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'trades': trades
        }
        
        return results
    
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest model."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance - {self.ticker}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [self.features[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved as '{self.ticker}_feature_importance.png'")
        
        # Print feature importance
        print("\nFeature Importance:")
        for i in indices:
            print(f"  {self.features[i]}: {importances[i]:.4f}")
    
    def plot_candlestick_with_signals(self):
        """Create interactive candlestick chart with Buy/Hold/Sell markers."""
        print("Creating candlestick chart...")
        
        # Filter only buy and sell signals for markers
        buy_signals = self.data[self.data['Predicted_Signal'] == 1]
        sell_signals = self.data[self.data['Predicted_Signal'] == -1]
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{self.ticker} - Candlestick Chart with Trading Signals', 'Portfolio Value')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Buy signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.98,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
        
        # Sell signals
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.02,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Portfolio_Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Add horizontal line for initial capital
        fig.add_hline(
            y=self.initial_capital,
            line_dash="dash",
            line_color="gray",
            row=2, col=1,
            annotation_text=f"Initial: ${self.initial_capital}"
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        # Save chart
        chart_filename = f'{self.ticker}_trading_chart.html'
        fig.write_html(chart_filename)
        print(f"Interactive chart saved as '{chart_filename}'")
        
        return fig
    
    def run(self):
        """Execute the complete trading algorithm pipeline."""
        print("="*60)
        print(f"SWING TRADING ALGORITHM - {self.ticker}")
        print("="*60)
        
        # Step 1: Fetch data
        self.fetch_data()
        
        # Step 2: Compute technical indicators
        self.compute_technical_indicators()
        
        # Step 3: Define labels
        self.define_labels()
        
        # Step 4: Train model
        accuracy = self.train_model()
        
        # Step 5: Predict signals
        self.predict_signals()
        
        # Step 6: Backtest
        results = self.backtest()
        
        # Step 7: Visualize
        self.plot_candlestick_with_signals()
        self.plot_feature_importance()
        
        # Print results
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Ticker: {self.ticker}")
        print(f"Time Period: {self.years} years ({len(self.data)} trading days)")
        print(f"Model Accuracy: {accuracy:.2%}")
        print(f"\nInitial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return']:+.2f}%")
        print(f"\nTotal Trades: {results['total_trades']}")
        print(f"  - Buy Trades: {results['buy_trades']}")
        print(f"  - Sell Trades: {results['sell_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print("="*60)
        
        return results


def main():
    """Main function to run the swing trading algorithm."""
    # Example usage
    print("Swing Trading Algorithm with Random Forest\n")
    
    # Get user input or use defaults
    ticker = input("Enter stock ticker (default: AAPL): ").strip().upper() or "AAPL"
    years_input = input("Enter number of years (1-10, default: 2): ").strip()
    years = int(years_input) if years_input else 2
    
    # Validate years
    if years < 1 or years > 10:
        print("Invalid years. Using default of 2 years.")
        years = 2
    
    # Create and run algorithm
    algo = SwingTradingAlgorithm(ticker=ticker, years=years)
    results = algo.run()
    
    return algo, results


if __name__ == "__main__":
    algo, results = main()


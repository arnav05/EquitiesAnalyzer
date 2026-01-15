# Swing Trading Algorithm - Usage Guide

## Quick Start

### 1. Create Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated from your global Python installation.

**Windows (PowerShell):**
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: Make sure your virtual environment is activated (you should see `(venv)` in your prompt) before installing or running the algorithm.

### 2. Run the Algorithm

#### Interactive Mode (Recommended for first-time users)

```bash
python swing_trading_algorithm.py
```

You'll be prompted to enter:
- **Stock ticker**: e.g., AAPL, MSFT, GOOGL, TSLA, AMZN
- **Number of years**: 1, 2, or 3 years of historical data

#### Command-Line Example

```bash
python example.py
```

This runs a pre-configured example with AAPL for 2 years.

#### Quick Test

```bash
python quick_test.py
```

Runs a quick 1-year test to verify everything is working correctly.

## Understanding the Output

### Console Output

The algorithm prints several sections:

1. **Data Fetching**: Shows how many days of data were downloaded
2. **Technical Indicators**: Confirms indicators were calculated
3. **Signal Distribution**: Shows how many Buy/Hold/Sell signals were generated
4. **Model Training**: Displays model accuracy on training data
5. **Backtesting**: Simulates trading with $1000
6. **Performance Summary**: Final results including:
   - Initial and final portfolio values
   - Total return percentage
   - Number of trades
   - Win rate

### Generated Files

After running, you'll find two new files:

1. **`{TICKER}_trading_chart.html`**
   - Interactive candlestick chart
   - Green triangles = Buy signals
   - Red triangles = Sell signals
   - Bottom panel shows portfolio value over time
   - Hover over any point for detailed information

2. **`{TICKER}_feature_importance.png`**
   - Bar chart showing which indicators were most useful
   - Higher bars = more important for predictions

## How to Use the Results

### Interpreting Model Accuracy

- **> 70%**: Good performance, model found meaningful patterns
- **60-70%**: Moderate performance, results may vary
- **< 60%**: Poor performance, consider different parameters or indicators

### Interpreting Win Rate

- **> 60%**: Strong win rate, more profitable trades than losses
- **50-60%**: Moderate win rate, roughly balanced
- **< 50%**: Weak win rate, losing more often than winning

### Understanding Returns

The total return shows how much profit/loss you would have made:
- **Positive return**: Algorithm was profitable
- **Negative return**: Algorithm lost money
- **Compare to buy-and-hold**: Check if returns beat simply holding the stock

## Advanced Usage

### Programmatic Usage

Create custom scripts to analyze multiple stocks or different parameters:

```python
from swing_trading_algorithm import SwingTradingAlgorithm

# Create algorithm instance
algo = SwingTradingAlgorithm(ticker="AAPL", years=2)

# Run individual steps
algo.fetch_data()
algo.compute_technical_indicators()
algo.define_labels()
algo.train_model()
algo.predict_signals()

# Backtest and get results
results = algo.backtest()

# Access the data
print(algo.data.head())
print(algo.data['RSI'].describe())

# Access model details
print(f"Feature importances: {algo.model.feature_importances_}")

# Create visualizations
algo.plot_candlestick_with_signals()
algo.plot_feature_importance()
```

### Customizing Parameters

Edit `swing_trading_algorithm.py` to modify:

#### Win Condition Threshold
```python
# In the run() method, modify this line:
self.define_labels(forward_days=3, threshold=0.02)  # 2% instead of 1%
```

#### Initial Capital
```python
# In __init__ method:
self.initial_capital = 5000  # Start with $5000 instead of $1000
```

#### Technical Indicator Periods
```python
# In compute_technical_indicators method:
self.data['RSI'] = self.calculate_rsi(period=20)  # 20-day RSI instead of 14
```

#### Random Forest Parameters
```python
# In train_model method:
self.model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,      # Deeper trees
    random_state=42,
    n_jobs=-1
)
```

## Comparing Multiple Stocks

Use the example script to compare different stocks:

```python
from swing_trading_algorithm import SwingTradingAlgorithm

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
results = []

for ticker in tickers:
    algo = SwingTradingAlgorithm(ticker=ticker, years=2)
    result = algo.run()
    results.append({
        'ticker': ticker,
        'return': result['total_return'],
        'win_rate': result['win_rate']
    })

# Find best performer
best = max(results, key=lambda x: x['return'])
print(f"Best performer: {best['ticker']} with {best['return']:.2f}% return")
```

## Tips for Better Results

1. **Use 2-3 years of data**: More data = better patterns
2. **Test multiple stocks**: Some stocks are more predictable
3. **Check feature importance**: Focus on which indicators matter most
4. **Compare to baseline**: Calculate buy-and-hold return to compare
5. **Look at win rate**: High win rate with low returns may indicate small gains

## Troubleshooting

### "No data found for ticker"
- Check if the ticker symbol is correct
- Try a more popular ticker (AAPL, MSFT, etc.)
- Check your internet connection

### "Not enough data after cleaning"
- Try using more years of data
- The stock might be too new or have missing data

### Poor performance (< 50% accuracy)
- Stock might be too volatile or random
- Try different parameters (threshold, indicators)
- Some stocks are harder to predict than others

### Installation errors
- Make sure you have Python 3.8+
- Use a virtual environment: `python -m venv venv`
- Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
- Then install: `pip install -r requirements.txt`

## Next Steps

1. **Backtest on historical data**: The current implementation trains and tests on the same data (overfitting risk)
2. **Add train/test split**: Split data into training and validation sets
3. **Implement walk-forward analysis**: Test on unseen future data
4. **Add risk management**: Stop-loss, position sizing, portfolio diversification
5. **Try other indicators**: Volume profiles, Fibonacci levels, support/resistance
6. **Experiment with other models**: XGBoost, Neural Networks, LSTM

## Important Disclaimer

⚠️ **This is for educational purposes only.**

- Past performance does NOT guarantee future results
- The algorithm trains on historical data which may not represent future market conditions
- Real trading involves transaction costs, slippage, and market impact not modeled here
- Always paper trade first and never risk money you can't afford to lose
- Consult with financial professionals before making investment decisions

## Support

For issues or questions:
1. Check the main README.md
2. Review this usage guide
3. Run the quick_test.py to verify setup
4. Check the console output for error messages

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Quick test
python quick_test.py

# 3. Run example
python example.py

# 4. View results
# Open {TICKER}_trading_chart.html in your browser
# View {TICKER}_feature_importance.png

# 5. Try different stocks
python swing_trading_algorithm.py
# Enter: MSFT
# Enter: 2

# 6. Analyze results and iterate
```

Happy Trading! 📈


"""
Example usage of the Swing Trading Algorithm
"""

from swing_trading_algorithm import SwingTradingAlgorithm


def run_example():
    """Run example trading algorithm for AAPL."""
    print("Running example swing trading algorithm for AAPL (2 years)...\n")
    
    # Create algorithm instance
    algo = SwingTradingAlgorithm(ticker="AAPL", years=2)
    
    # Run the complete pipeline
    results = algo.run()
    
    # You can access the data and model for further analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Show last 5 predictions
    print("\nLast 5 Trading Days:")
    print(algo.data[['Close', 'RSI', 'MACD', 'Predicted_Signal']].tail())
    
    # Show trade history (first 5 and last 5)
    if len(results['trades']) > 0:
        print("\nFirst 5 Trades:")
        for trade in results['trades'][:5]:
            print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} "
                  f"{trade['shares']:.2f} shares at ${trade['price']:.2f}")
        
        if len(results['trades']) > 5:
            print("\n  ...")
            print("\nLast 5 Trades:")
            for trade in results['trades'][-5:]:
                print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} "
                      f"{trade['shares']:.2f} shares at ${trade['price']:.2f}")
    
    return algo, results


def run_multiple_tickers():
    """Run algorithm for multiple tickers and compare results."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results_summary = []
    
    print("\nRunning algorithm for multiple tickers...\n")
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            algo = SwingTradingAlgorithm(ticker=ticker, years=2)
            results = algo.run()
            results_summary.append({
                'ticker': ticker,
                'return': results['total_return'],
                'win_rate': results['win_rate'],
                'trades': results['total_trades']
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Ticker':<10} {'Return':<15} {'Win Rate':<15} {'Trades':<10}")
    print("-"*60)
    for result in results_summary:
        print(f"{result['ticker']:<10} {result['return']:+.2f}%{'':<9} "
              f"{result['win_rate']:.2f}%{'':<9} {result['trades']:<10}")
    print("="*60)


if __name__ == "__main__":
    # Run single example
    algo, results = run_example()
    
    # Uncomment to run multiple tickers comparison
    # run_multiple_tickers()


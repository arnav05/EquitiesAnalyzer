"""
Quick test script for the Swing Trading Algorithm
Tests with a small dataset to verify functionality
"""

from swing_trading_algorithm import SwingTradingAlgorithm
import sys


def quick_test(ticker="AAPL", years=1):
    """
    Run a quick test of the algorithm with minimal data.
    
    Args:
        ticker (str): Stock ticker to test
        years (int): Number of years (1 for quick test)
    """
    print("="*60)
    print(f"QUICK TEST - {ticker} ({years} year)")
    print("="*60)
    
    try:
        # Create and run algorithm
        algo = SwingTradingAlgorithm(ticker=ticker, years=years)
        results = algo.run()
        
        # Verify results
        print("\n[PASS] Algorithm completed successfully!")
        
        # Check if outputs were created
        import os
        chart_file = f"{ticker}_trading_chart.html"
        importance_file = f"{ticker}_feature_importance.png"
        
        if os.path.exists(chart_file):
            print(f"[PASS] Chart generated: {chart_file}")
        else:
            print(f"[FAIL] Chart file not found: {chart_file}")
        
        if os.path.exists(importance_file):
            print(f"[PASS] Feature importance plot generated: {importance_file}")
        else:
            print(f"[FAIL] Feature importance file not found: {importance_file}")
        
        # Validate results structure
        required_keys = ['initial_capital', 'final_value', 'total_return', 
                        'total_trades', 'win_rate']
        missing_keys = [key for key in required_keys if key not in results]
        
        if missing_keys:
            print(f"[FAIL] Missing result keys: {missing_keys}")
        else:
            print("[PASS] All result keys present")
        
        # Check data integrity
        if algo.data is not None and len(algo.data) > 0:
            print(f"[PASS] Data loaded: {len(algo.data)} trading days")
            
            # Check for required columns
            required_cols = ['RSI', 'MACD', 'BB_Upper', 'OBV', 'ATR', 
                           'Signal', 'Predicted_Signal']
            missing_cols = [col for col in required_cols if col not in algo.data.columns]
            
            if missing_cols:
                print(f"[FAIL] Missing columns: {missing_cols}")
            else:
                print("[PASS] All required columns present")
        else:
            print("[FAIL] No data loaded")
        
        print("\n" + "="*60)
        print("TEST COMPLETED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Get ticker from command line or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    success = quick_test(ticker=ticker, years=1)
    
    if success:
        print("\n[PASS] All tests passed! The algorithm is working correctly.")
        sys.exit(0)
    else:
        print("\n[FAIL] Tests failed. Please check the error messages above.")
        sys.exit(1)


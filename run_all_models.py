"""
Master Script - Run all models in sequence
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:")
        print(result.stderr)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully!")
        return True
    else:
        print(f"\n✗ {description} failed with error code {result.returncode}")
        return False

def main():
    print("=" * 80)
    print("TRANSFERIQ - COMPLETE MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Check if data exists
    if not os.path.exists('player_data.csv'):
        print("ERROR: player_data.csv not found!")
        print("Please make sure the data file is in the current directory.")
        return
    
    print("✓ Data file found")
    
    # List of scripts to run
    scripts = [
        ('models/lstm_univariate.py', 'Univariate LSTM'),
        ('models/lstm_multivariate.py', 'Multivariate LSTM'),
        ('models/lstm_encoder_decoder.py', 'Encoder-Decoder LSTM'),
        ('models/create_stacking_dataset.py', 'Create Stacking Dataset'),
        ('models/train_xgboost_stacking.py', 'Train XGBoost Stacking Model'),
    ]
    
    # Run each script
    for script, description in scripts:
        if not run_script(script, description):
            print(f"\nPipeline stopped at: {description}")
            return
    
    print("\n" + "=" * 80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start Flask backend: cd Userinterface/backend && python api.py")
    print("2. Start React frontend: cd Userinterface/frontend && npm start")
    print("3. Open browser to http://localhost:3000")

if __name__ == "__main__":
    main()
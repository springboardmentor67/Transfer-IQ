import subprocess
import sys
import os

def run_script(script_info):
    if isinstance(script_info, str):
        script_name = script_info
        args = []
    else:
        script_name = script_info[0]
        args = script_info[1:]

    script_path = os.path.join("scripts", script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found.")
        return False
    
    print(f"Running {script_name} {' '.join(args)}...")
    try:
        cmd = [sys.executable, script_path] + args
        result = subprocess.run(cmd, check=True)
        print(f"Successfully ran {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    # Define the sequence of scripts to run
    # Omitting 'week1_transfermarkt_scrape.py' as it is long-running/risky. 
    # Ensure data exists or use mock/open data if possible.
    
    scripts = [
        ["week1_download_statsbomb.py", "--competition-id", "43", "--season-id", "106"], # World Cup 2022
        "week1_build_player_performance.py",
        ["week1_twitter_sentiment_mock.py", "--input", "data/processed/statsbomb_player_performance.csv"], # Ensure input is provided
        "week2_data_processing.py",
        "week3_sentiment_pipeline.py",
        "week4_prep_timeseries.py",
        "week4_generate_sequences.py",
        "week5_lstm_model.py",
        "week6_ensemble_model.py",
        "week7_evaluation.py"
    ]
    
    print("Starting Project Pipeline...")
    for script_info in scripts:
        success = run_script(script_info)
        if not success:
            if isinstance(script_info, str):
                name = script_info
            else:
                name = script_info[0]
            print(f"Pipeline stopped due to error in {name}")
            sys.exit(1)
            
    print("\nProject Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete pipeline runner for Titanic project
Runs data cleaning, model training, and optionally starts the web applications
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("❌ Error!")
        if result.stderr:
            print("Error:", result.stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Titanic ML Pipeline")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip model training (use existing model)")
    parser.add_argument("--start-streamlit", action="store_true",
                       help="Start Streamlit app after training")
    parser.add_argument("--start-fastapi", action="store_true",
                       help="Start FastAPI app after training")
    parser.add_argument("--docker", action="store_true",
                       help="Use Docker to run applications")
    
    args = parser.parse_args()
    
    # Change to the python_version directory
    original_dir = Path.cwd()
    script_dir = Path(__file__).parent
    if script_dir.name != "titanic-python":
        print("Please run this script from the python_version directory")
        sys.exit(1)
    
    print("🚢 Titanic ML Pipeline Runner")
    print(f"Working directory: {Path.cwd()}")
    
    # Step 1: Install requirements
    if not run_command("uv add -r requirements.txt", "Installing Python requirements"):
        print("Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Step 3: Copy data if needed
    data_source = "../data/train.csv"
    if Path(data_source).exists():
        run_command(f"cp {data_source} data/", "Copying training data")
    
    test_data_source = "../data/test.csv"
    if Path(test_data_source).exists():
        run_command(f"cp {test_data_source} data/", "Copying test data")
    
    # Step 4: Run data cleaning
    if not run_command("python data_cleaning.py", "Data cleaning and preprocessing"):
        print("Data cleaning failed!")
        return
    
    # Step 5: Run model training (unless skipped)
    if not args.skip_training:
        if not run_command("python model_training.py", "Model training"):
            print("Model training failed!")
            return
    else:
        print("⏭️ Skipping model training")
    
    # Step 6: Start applications if requested
    if args.docker:
        if args.start_streamlit or args.start_fastapi:
            print("\n🐳 Starting applications with Docker...")
            run_command("docker-compose up --build -d", "Starting Docker containers")
            print("\n📱 Applications started!")
            print("- Streamlit app: http://localhost:8501")
            print("- FastAPI app: http://localhost:8000")
            print("- FastAPI docs: http://localhost:8000/docs")
    else:
        if args.start_streamlit:
            print("\n🎨 Starting Streamlit application...")
            print("Navigate to: http://localhost:8501")
            run_command("streamlit run streamlit_app.py", "Starting Streamlit")
        
        if args.start_fastapi:
            print("\n🚀 Starting FastAPI application...")
            print("Navigate to: http://localhost:8000")
            print("API docs at: http://localhost:8000/docs")
            run_command("uvicorn fastapi_app:app --host 0.0.0.0 --port 8000", "Starting FastAPI")
    
    print("\n✅ Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. To run Streamlit: streamlit run streamlit_app.py")
    print("2. To run FastAPI: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000")
    print("3. To run with Docker: docker-compose up --build")

if __name__ == "__main__":
    main()
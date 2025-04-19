import argparse
import os
import sys
import subprocess

def main():
    """
    Script to run the MathFi application with various options.
    """
    parser = argparse.ArgumentParser(description='Run the MathFi handwritten math solver')
    parser.add_argument('--train', action='store_true', help='Train the model before running the app')
    parser.add_argument('--port', type=int, default=8501, help='Port to run the Streamlit app on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open a browser window')
    
    args = parser.parse_args()
    
    # Check if models directory exists, create if not
    if not os.path.exists('models/symbol_recognition_model'):
        os.makedirs('models/symbol_recognition_model', exist_ok=True)
        print("Created models directory")
    
    # Train the model if requested
    if args.train:
        print("Training the model...")
        subprocess.run([sys.executable, 'train_model.py'])
    
    # Run the Streamlit app
    print(f"Starting the MathFi app on port {args.port}...")
    cmd = [
        'streamlit', 'run', 'app.py',
        '--server.port', str(args.port)
    ]
    
    if args.no_browser:
        cmd.extend(['--server.headless', 'true'])
    
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 
import os
import subprocess

def init_dvc():
    """Initialize DVC for data versioning"""
    if not os.path.exists(".dvc"):
        subprocess.run(["dvc", "init"], check=True)
        subprocess.run(["git", "add", ".dvc"], check=True)
        subprocess.run(["git", "commit", "-m", "Initialize DVC"], check=True)
        print("DVC initialized")
    else:
        print("DVC already initialized")

def track_data(file_path):
    """Track a data file with DVC"""
    subprocess.run(["dvc", "add", file_path], check=True)
    dvc_file = f"{file_path}.dvc"
    subprocess.run(["git", "add", dvc_file], check=True)
    print(f"Tracking {file_path} with DVC")

def create_pipeline():
    """Create DVC pipeline stages"""
    stages = [
        ("data_preprocessing", "python preprocess.py"),
        ("train", "python train.py"),
        ("evaluate", "python evaluate.py")
    ]
    
    for name, command in stages:
        subprocess.run(["dvc", "run", "-n", name, "-d", command.split()[1], command], check=True)

def push_to_remote(remote_name="storage"):
    """Push data to DVC remote"""
    subprocess.run(["dvc", "remote", "add", "-d", remote_name, "/path/to/storage"], check=False)
    subprocess.run(["dvc", "push"], check=True)
    print("Data pushed to remote")

def pull_from_remote():
    """Pull data from DVC remote"""
    subprocess.run(["dvc", "pull"], check=True)
    print("Data pulled from remote")

if __name__ == "__main__":
    print("=== DVC Data Versioning ===")
    print("To use DVC:")
    print("1. dvc init")
    print("2. dvc add data.csv")
    print("3. git add data.csv.dvc")
    print("4. dvc push")
    print("5. dvc pull")

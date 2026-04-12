import os
import subprocess
import sys


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(project_root, "requirements.txt")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

    required_dirs = [
        os.path.join(project_root, "model", "saved"),
        os.path.join(project_root, "model", "train_test", "results"),
        os.path.join(project_root, "data", "states"),
    ]

    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)

    print("Setup complete. Ready to run the pipeline.")


if __name__ == "__main__":
    main()

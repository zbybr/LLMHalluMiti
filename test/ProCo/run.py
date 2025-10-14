import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Wrapper to run ProCo comparison pipelines")
    parser.add_argument('--dataset_csv', required=True, help="Path to the local dataset CSV")
    parser.add_argument('--model_key', required=True, help="Model key for Advanced_ProCo")
    args = parser.parse_args()

    # Build the command to run main.py with user parameters
    command = [
        "python", "main.py",
        "--dataset_csv", args.dataset_csv,
        "--model_key", args.model_key
    ]

    # Execute the main pipeline
    subprocess.run(command)


if __name__ == "__main__":
    main()

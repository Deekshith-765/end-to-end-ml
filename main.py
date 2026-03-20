import subprocess

def run_step(step_name, command):
    print(f"\n=== {step_name} ===")

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("Error:", result.stderr)
        exit(1)

# Run pipeline
run_step("Step 1: Detecting drift", ["python", "detect.py"])
run_step("Step 2: Retraining if needed", ["python", "retrain.py"])

print("\nPipeline completed successfully!")

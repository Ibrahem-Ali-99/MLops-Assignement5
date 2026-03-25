import mlflow
import sys
import os

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics["accuracy"]
print(f"Accuracy: {accuracy:.4f}")

THRESHOLD = 0.85

if accuracy < THRESHOLD:
    print(f"FAILED: {accuracy:.4f} is below threshold {THRESHOLD}. Blocking deployment.")
    sys.exit(1)
else:
    print(f"PASSED: {accuracy:.4f} meets threshold {THRESHOLD}. Proceeding to deploy.")
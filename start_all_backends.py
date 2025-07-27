import subprocess
import sys
import time

# List of backend scripts to start
processes = [
    [sys.executable, "product_workbench_backlog_management.py"],
    [sys.executable, "product_workbench_requirement_definition.py"],
    [sys.executable, "poc3/backend/app.py"]
    # Add more scripts here if needed
]

procs = []
try:
    for cmd in processes:
        print(f"Starting: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd))
    print("All backends started. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping all backends...")
    for proc in procs:
        proc.terminate()
    sys.exit(0)

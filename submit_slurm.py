import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--script_path", type=str, required=True)
args = parser.parse_args()

# Submit the SLURM script
result = subprocess.run(["sbatch", args.script_path], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("Error submitting job:", result.stderr)

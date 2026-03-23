import subprocess
import os

print("=========================================================")
print(" Running Reliability Simulations (Analytical & Monte Carlo)")
print("=========================================================")

# Ensure results directories exist
os.makedirs(os.path.join("results", "edac_uber"), exist_ok=True)

# 1. Analytical - Random Mode (independent bit flips)
print("\n>>> Run 1/4: Analytical - Random SEU Mode")
subprocess.run(["python", "ecc_simulator.py", "--analytical", "--mode", "random"], check=True)

# 2. Analytical - Burst Mode (LEO radiation model)
print("\n>>> Run 2/4: Analytical - Burst SEU Mode")
subprocess.run(["python", "ecc_simulator.py", "--analytical", "--mode", "burst"], check=True)

# 3. Monte Carlo - Random Mode
print("\n>>> Run 3/4: Monte Carlo - Random SEU Mode")
subprocess.run(["python", "ecc_simulator.py", "--monte-carlo", "--mode", "random"], check=True)

# 4. Monte Carlo - Burst Mode
print("\n>>> Run 4/4: Monte Carlo - Burst SEU Mode")
subprocess.run(["python", "ecc_simulator.py", "--monte-carlo", "--mode", "burst"], check=True)

print("\n=========================================================")
print(" All reliability benchmarks complete!")
print(" Plots saved in: results/edac_uber/")
print("=========================================================\n")

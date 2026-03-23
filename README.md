# NAND Error Correction Analysis

This repository contains the simulation tools and timing scripts created for investigating 
NAND flash error correction (EDAC) architectures under various error models (including LEO burst models).

There are two main sets of plots generated from this codebase: EDAC simulation/analytical plots, and STM32 latency/throughput charts.

## How to Run (Plot Generators)

These scripts orchestrate the execution and export images to the `results/` directory.

- `python run_scripts/run_edac_plots.py`  
  Runs the main UBER (Uncorrectable Bit Error Rate) evaluation. Generates plots for the random (independent) fault model and compound Poisson LEO burst models. Both analytical and full Monte-Carlo simulations are generated. Outputs are placed in `results/edac_uber/`.

- `python run_scripts/run_stm32_plots.py`  
  Generates the encode/decode throughput and latency charts based on the STM32H753 hardware parameters and the physical NAND delays. Outputs are placed in `results/timing/`.


## Core Modules & Utilities

The simulation logic and models reside in the following libraries:

- **`ecc_utils.py`**: Mathematical operations, codec wrappers (RS, BCH, LDPC mapping), Galois Field calculations, and generic helper methods.
- **`fault_injection.py`**: Defines the fault injection models (e.g., LEO single-bit vs burst distributions). Contains the Monte-Carlo simulation logic and Poisson sampling steps.
- **`ecc_simulator.py`**: Main configurable script for calculating UBER limits, performing density evolution approximations, and structuring the matplotlib subplots for EDAC schemes.
- **`lib_timing_base.py`**: Cycle-based ECC timing model for RS, BCH, and LDPC architectures.
- **`lib_timing_stm32.py`**: STM32H753-specific timing overlay, injecting memory hierarchy costs onto `lib_timing_base` parameters.

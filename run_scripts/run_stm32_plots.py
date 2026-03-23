"""
Script to evaluate stm32 execution and compute throughputs considering
the underlying NAND device delays.
"""

import os
import matplotlib.pyplot as plt
import lib_timing_stm32 as stm32_env
from lib_timing_base import build_all_timings, InfeasibleEntry

NAND_DEVICES = {
    "MT29F256G08": {"desc": "MT29F256G08AUCABPB-10ITZ:A", "page_size": 8640, "read_us": 208, "program_us": 523, "erase_us": 1500.0},
    "MT29F512G08": {"desc": "MT29F512G08CUAAAC5:A", "page_size": 8640, "read_us": 248, "program_us": 1473, "erase_us": 3800.0},
    "3DFN128G08":  {"desc": "3DFN128G08US8761", "page_size": 4224, "read_us": 133, "program_us": 335, "erase_us": 700.0},
}

def plot_device_throughputs(freq_hz: float, worst_case: bool, output_dir: str = "results/timing"):
    os.makedirs(output_dir, exist_ok=True)
    
    for dev_name, info in NAND_DEVICES.items():
        page_size = info["page_size"]
        timings = build_all_timings(page_size, worst_case=worst_case)
        valid = [t for t in timings if not isinstance(t, InfeasibleEntry)]
        
        if not valid:
            continue
            
        labels = [t.label for t in valid]
        enc_tp = [t.enc_throughput_mbits(freq_hz, info["program_us"]) for t in valid]
        dec_tp = [t.dec_throughput_mbits(freq_hz, info["read_us"]) for t in valid]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], enc_tp, width=0.4, label="Program + Encode", color="#228833")
        ax.bar([i + 0.2 for i in x], dec_tp, width=0.4, label="Read + Decode", color="#CCBB44")
        
        ax.set_ylabel("Throughput (Mbit/s)")
        ax.set_title(f"Throughput {dev_name} (Page: {page_size}B) @ {freq_hz/1e6:.1f}MHz")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"throughput_{dev_name}.png"))
        plt.close()
        print(f"Saved: {output_dir}/throughput_{dev_name}.png")

if __name__ == "__main__":
    print("Generating NAND-specific throughput charts with STM32 timings...")
    plot_device_throughputs(freq_hz=stm32_env.STM32H753_FREQ_MAX_HZ, worst_case=True)
    print("Done!")

import os
import matplotlib.pyplot as plt
import numpy as np
import re

# Define the data files and their corresponding temperatures
data_files = ["T_0.3_m4", "T_0.2_m4", "T_0.1_m4", "T_0.06_m4"]
temperatures = ["0.3", "0.2", "0.1", "0.06"]

# Create a figure with 4 subplots arranged in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()  # Flatten the 2D array to 1D for easy iteration

# Color palette for the four chi components
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
markers = ["o", "s", "^", "d"]

for idx, file_name in enumerate(data_files):
    # Check if file exists before reading
    if not os.path.exists(file_name):
        print(f"Warning: File {file_name} not found. Skipping...")
        axs[idx].text(
            0.5,
            0.5,
            f"Missing {file_name}",
            ha="center",
            va="center",
            fontsize=14,
        )
        continue

    # Load data: Column 0 = t, Columns 1-4 = chi_0, chi_1, chi_2, chi_3
    try:
        data = np.loadtxt(file_name)
        t = data[:, 0]

        # Plot each chi_i
        for i in range(4):
            axs[idx].plot(
                t,
                data[:, i + 1],
                label=r"$\chi_{" + str(i) + "}$",
                color=colors[i],
                marker=markers[i],
                markersize=4,
                linewidth=1.5,
                markevery=max(1, len(t) // 20),  # Avoid crowded markers
            )

        # Subplot styling
        axs[idx].set_title(r"$k_{\rm B}T ="+ temperatures[idx]+"\Gamma$", fontsize=14)
        axs[idx].grid(True, linestyle="--", alpha=0.6)
        axs[idx].legend(fontsize=11, loc="best")
        axs[idx].tick_params(axis="both", labelsize=12)
        if data[-1, 3]/data[-1, 0] > 50 :
            axs[idx].set_yscale("log")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Set shared x-labels for the bottom row and y-labels for the left column
for ax in axs[2:]:
    ax.set_xlabel(r"$\Gamma\ t$", fontsize=13)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel(r"$\chi_i$", fontsize=13)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save and show the plot
plt.savefig("chi_evolution_vs_temperature_m4.png", dpi=300)
plt.show()
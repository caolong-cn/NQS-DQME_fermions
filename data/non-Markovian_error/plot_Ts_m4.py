import os
import matplotlib.pyplot as plt
import numpy as np
import re


class latexmath:
    def __enter__(self):
        plt.rcParams['mathtext.fontset'] = 'cm'
        return 0
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        plt.rcParams['mathtext.fontset'] = 'custom'
        return 0

fname = 'Arial'
fname_italic = 'Arial:italic'
fname2 = 'Arial'

plt.rcParams['font.family'] = fname  #使得坐标轴刻度标签字体变化
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = fname
plt.rcParams['mathtext.it'] = fname_italic

fig = plt.figure(figsize=(12,10))

dy = 0.37
y1 = 0.095
y2 = 2*y1+dy

x1 = 0.085
x2 = 0.55
dx = 0.39
ax_a = fig.add_axes([x1,y2,dx,dy])
# ax_a.axis('off')
ax_a.text(-0.08,1.05,r'${\bf (a)}$',fontdict={'family':fname2,'size': 16},transform=ax_a.transAxes)

# ax_c = fig.add_axes([0.09,0.40,0.87,0.265])
# ax_b.text(-0.04,1.05,'(b)',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)
ax_b = fig.add_axes([x2,y2,dx,dy])
ax_b.text(-0.08,1.05,r'${\bf (b)}$',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)

ax_c = fig.add_axes([x1,y1,dx,dy])
ax_c.text(-0.08,1.05,r'${\bf (c)}$',fontdict={'family':fname2,'size': 16},transform=ax_c.transAxes)

ax_d = fig.add_axes([x2,y1,dx,dy])
ax_d.text(-0.08,1.05,r'${\bf (d)}$',fontdict={'family':fname2,'size': 16},transform=ax_d.transAxes)

axs = [ax_a,ax_b,ax_c,ax_d]

# Color palette for the four chi components
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
markers = ["o", "s", "^", "d"]


# Define the data files and their corresponding temperatures
data_files = ["T_0.3_m4", "T_0.2_m4", "T_0.1_m4", "T_0.06_m4"]
temperatures = ["3.0", "2.0", "1.0", "0.6"]



for idx, file_name in enumerate(data_files):
    print(idx)
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
        axs[idx].set_title(r"$k_{\rm B}T ="+ temperatures[idx]+"\Gamma$", fontsize=15)
        # axs[idx].grid(True, linestyle="--", alpha=0.6)
        if idx==0:
            axs[idx].legend(fontsize=10, loc=(0.5,0.5))
        else:
            axs[idx].legend(fontsize=11, loc="best")
        axs[idx].tick_params(axis="both", labelsize=12)
        if data[-1, 3]/data[-1, 0] > 20 :
            axs[idx].set_yscale("log")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Set shared x-labels for the bottom row and y-labels for the left column
for ax in axs[2:]:
    ax.set_xlabel(r'$t$ ($1/\Gamma$)', fontsize=15)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel(r"$\chi_m$ ($1/\Gamma$)", fontsize=15)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save and show the plot
plt.savefig("chi_evolution_vs_temperature_m4.png", dpi=300)
plt.savefig("chi_evolution_vs_temperature_m4.pdf")

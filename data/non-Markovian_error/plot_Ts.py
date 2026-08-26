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

fig = plt.figure(figsize=(8.6,10.3))

ax_a = fig.add_axes([0.095,0.71,0.855,0.25])
ax_a.axis('off')
ax_a.text(-0.04,1.00,r'${\bf (a)}$',fontdict={'family':fname2,'size': 16},transform=ax_a.transAxes)

x1 = 0.095
x2 = 0.55
dx = 0.39

y1 = 0.056
y2 = 0.42
dy = 0.3
# ax_c = fig.add_axes([0.09,0.40,0.87,0.265])
# ax_b.text(-0.04,1.05,'(b)',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)
ax_b = fig.add_axes([x1,y2,dx,dy])
ax_b.text(-0.08,1.05,r'${\bf (b)}$',fontdict={'family':fname2,'size': 16},transform=ax_b.transAxes)

ax_c = fig.add_axes([x2,y2,dx,dy])
ax_c.text(-0.08,1.05,r'${\bf (c)}$',fontdict={'family':fname2,'size': 16},transform=ax_c.transAxes)

ax_d = fig.add_axes([x1,y1,dx,dy])
ax_d.text(-0.08,1.05,r'${\bf (d)}$',fontdict={'family':fname2,'size': 16},transform=ax_d.transAxes)

ax_e = fig.add_axes([x2,y1,dx,dy])
ax_e.text(-0.08,1.05,r'${\bf (e)}$',fontdict={'family':fname2,'size': 16},transform=ax_e.transAxes)

axs = [ax_b,ax_c,ax_d,ax_e]

# Color palette for the four chi components
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
markers = ["o", "s", "^", "d"]


# Define the data files and their corresponding temperatures
data_files = ["T_0.3", "T_0.2", "T_0.1", "T_0.06"]
temperatures = ["3.0", "2.0", "1.0", "0.6"]



#case1示意图
# case1 = plt.imread('case1_a.png')
# ax_a.imshow(case1)#,cmap='hot'
# ax_a.set_xticks(())
# ax_a.set_yticks(())


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
        axs[idx].set_title(r"$k_{\rm B}T ="+ temperatures[idx]+"\Gamma$", fontsize=15)
        # axs[idx].grid(True, linestyle="--", alpha=0.6)
        if idx==0:
            axs[idx].legend(fontsize=12, loc=(0.5,0.5))
        elif idx==1:
            axs[idx].legend(fontsize=12, loc=(0.5,0.27))
        else:
            axs[idx].legend(fontsize=12, loc="lower right")
            # axs[idx].set_ylim(4e-2,5e3)
        axs[idx].tick_params(axis="both", labelsize=12)
        if data[-1, 3]/data[-1, 0] > 30 :
            axs[idx].set_yscale("log")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Set shared x-labels for the bottom row and y-labels for the left column
for ax in axs[2:]:
    ax.set_xlabel(r'$t$ ($1/\Gamma$)', fontsize=17)
for ax in [axs[0], axs[2]]:
    ax.set_ylabel(r"$\chi_m$ ($1/\Gamma$)", fontsize=17)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# Save and show the plot
plt.savefig("chi_evolution_vs_temperature.png", dpi=300)
plt.savefig("chi_evolution_vs_temperature.pdf")
plt.show()
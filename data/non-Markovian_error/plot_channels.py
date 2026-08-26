import numpy as np
import matplotlib.pyplot as plt

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

axes = [ax_a,ax_b,ax_c,ax_d]

# Raw data for each subfigure (as lists)
data = [
    [46.52974738, 10.9300578, 4.86999772, 2.344237],
    [155.50630766, 38.07710179, 16.64809536, 8.84623665, 3.33042773],
    [1292.71226263, 359.69129808, 175.9562467, 87.87760468, 27.70483632, 15.3308521],
    [5754.96157438, 1846.30874402, 962.66244974, 588.59355078, 300.43978292, 67.02442029, 48.4459338]
]

temperatures = ["3.0", "2.0", "1.0", "0.6"]
# Labels for each channel
channel_labels = [f'$\gamma_{i}$' for i in range(max(len(d) for d in data))]
# gammas = 
# channel_labels = [f'$50.0\,\Gamma$' for i in range(max(len(d) for d in data))]
# print(channel_labels)
for idx, ax in enumerate(axes):
    raw = np.array(data[idx])
    total = np.sum(raw)
    percentages = (raw / total) * 100
    
    # Use only labels that exist for this subfigure
    labels = channel_labels[:len(raw)]
    
    # Create bar plot
    ax.bar(labels, percentages, color="#1f77b4", edgecolor='black')
    
    # Add percentage labels on top of bars
    for i, pct in enumerate(percentages):
        ax.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=15)
    
    # Set title and labels
    # ax.set_title(f'Channel Distribution (Set {idx+1})')
    ax.set_title(r"$k_{\rm B}T ="+ temperatures[idx]+"\Gamma$", fontsize=17)

    ax.set_ylabel('Percentage (%)',fontsize=17)
    ax.set_xlabel('Channels',fontsize=17)
    
    # Set y-axis limit to give space for labels
    ax.set_ylim(0, 90)

# Adjust layout and display
# plt.tight_layout()
# plt.show()

# Optional: Save the figure
plt.savefig('channel_distribution.png', dpi=100)
plt.savefig("channel_distribution.pdf")
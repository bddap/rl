import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pat = re.compile(r'distance_m=([\d.]+)')
def load(p): return [float(m[1]) for m in map(pat.search, open(p)) if m]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for ax, arm, title, color in ((axes[0], 'ctrl', 'control: log-uniform spawn draw', '#4a6fa5'), (axes[1], 'uniform', 'treatment: uniform-distance draw', '#c46a2b')):
    pts = [load(f'traces/{arm}-s{t}-{s}.txt') for t in (351, 352, 353) for s in range(351, 371)]
    ax.scatter([d[0] for d in pts], [d[-1] for d in pts], s=18, color=color, alpha=0.8)
    ax.plot([0, 10], [0, 10], ls='--', color='#888', lw=1)
    ax.axhline(0.5, color='#3a9a4a', lw=1)
    ax.set_title(f'{title}\nmean policy, 60 renders')
    ax.set_xlabel('distance to ball at 1 s (m)')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
axes[0].set_ylabel('distance to ball at 20 s (m)')
fig.tight_layout()
fig.savefig('closing.png', dpi=110)

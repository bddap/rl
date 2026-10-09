import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pat = re.compile(r'distance_m=([\d.]+) .*upright=(-?[\d.]+)')
C = {'inv': '#eb6834', 'hold': '#1baf7a', 'up': '#2a78d6'}
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, dpi=110)
for col, (c, name) in enumerate([('40000512', '40,000,512 ticks'), ('48172032', '48,172,032 ticks')]):
    for s in range(351, 371):
        rows = [pat.search(l) for l in open(f'traces/{c}-{s}.txt')]
        rows = [(float(m[1]), float(m[2])) for m in rows if m]
        d, u = zip(*rows)
        t = range(1, 21)
        k = 20
        while k > 0 and d[k-1] <= 0.5: k -= 1
        cls = 'hold' if k <= 17 else ('inv' if u[-1] < 0 else 'up')
        kw = dict(color=C[cls], lw=2 if cls != 'up' else 1.2, alpha=1 if cls != 'up' else 0.55)
        axes[0][col].plot(t, u, **kw)
        axes[1][col].plot(t, d, **kw)
    axes[0][col].set_title(f'mean policy, seeds 351–370 — {name}', fontsize=11)
    axes[0][col].axhline(0, color='#999', lw=0.8)
    axes[1][col].axhline(0.5, color='#999', lw=0.8, ls='--')
    axes[1][col].set_xlabel('simulated seconds')
axes[0][0].set_ylabel('upright (carapace up · world Y)')
axes[1][0].set_ylabel('body-to-ball distance (m)')
for a in axes.flat:
    a.spines[['top', 'right']].set_visible(False); a.grid(alpha=0.2)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=C[k], lw=2) for k in ('inv', 'up', 'hold')],
           ['inverted at 20 s', 'upright, did not close and hold', 'closed ≤ 0.5 m and held'],
           loc='lower center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig('inversion-sweep.png')

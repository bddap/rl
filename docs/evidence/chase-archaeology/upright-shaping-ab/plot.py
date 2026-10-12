import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pat = re.compile(r'distance_m=([\d.]+) .*upright=(-?[\d.]+)')
ARM = {'ctrl': ('control, k = 0', '#7a7a7a'), 'k12': ('upright shaping, k = 12', '#2a78d6')}
fig = plt.figure(figsize=(13, 7.5), dpi=110)
ax = fig.add_subplot(2, 1, 1)
for arm, (name, col) in ARM.items():
    rows = [l.split() for l in open(f'{arm}-reach2m.txt')]
    ax.plot([float(r[0]) for r in rows], [float(r[1]) for r in rows], color=col, lw=2, marker='o', ms=3, label=name)
ax.axvspan(35, 40, color='#eb6834', alpha=0.08)
ax.text(37.5, 0.205, 'pre-registered\nreach bucket', ha='center', va='top', fontsize=9, color='#a8451f')
ax.set_xlabel('training ticks (M), 2M buckets'); ax.set_ylabel('train reach (episode-weighted)')
ax.legend(frameon=False, loc='upper left')
for i, (arm, (name, col)) in enumerate(ARM.items()):
    a = fig.add_subplot(2, 2, 3 + i)
    for s in range(351, 371):
        rows = [pat.search(l) for l in open(f'traces/{arm}-{s}.txt')]
        u = [float(m[2]) for m in rows if m]
        inv = u[-1] < 0
        a.plot(range(1, 21), u, color='#eb6834' if inv else col, lw=1.8 if inv else 1.1, alpha=1 if inv else 0.6)
    n = sum(1 for s in range(351, 371) if float(pat.findall(open(f'traces/{arm}-{s}.txt').read())[-1][1]) < 0)
    a.set_title(f'{name}: mean policy at 40M, seeds 351–370 — {n}/20 inverted at 20 s', fontsize=10)
    a.axhline(0, color='#999', lw=0.8); a.set_ylim(-1.05, 1.05); a.set_xlabel('simulated seconds')
    if i == 0: a.set_ylabel('upright')
for a in fig.axes:
    a.spines[['top', 'right']].set_visible(False); a.grid(alpha=0.2)
fig.tight_layout()
fig.savefig('upright-shaping-ab.png')

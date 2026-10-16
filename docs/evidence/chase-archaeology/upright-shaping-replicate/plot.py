import re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
bpat = re.compile(r'(\d+)-\s*(\d+)M\s+reach ([\d.]+)')
tpat = re.compile(r'distance_m=([\d.]+) .*upright=(-?[\d.]+)')
COL = {'ctrl': '#7a7a7a', 'k12': '#2a78d6'}
NAME = {'ctrl': 'control, k = 0', 'k12': 'upright shaping, k = 12'}
def buckets(path):
    rows = [bpat.search(l) for l in open(path)]
    return [(int(m[1]) + 2.5, float(m[3])) for m in rows if m and int(m[2]) <= 40]
def bucket_path(arm, seed):
    return f'../upright-shaping-ab/{arm}-buckets.txt' if seed == 351 else f'{arm}-s{seed}-buckets.txt'
def trace_path(arm, seed, s):
    return f'../upright-shaping-ab/traces/{arm}-{s}.txt' if seed == 351 else f'traces/{arm}-s{seed}-{s}.txt'
def sweep(arm, seed):
    inv = hold = 0
    for s in range(351, 371):
        rows = [tpat.search(l) for l in open(trace_path(arm, seed, s))]
        d = [float(m[1]) for m in rows if m]; u = [float(m[2]) for m in rows if m]
        inv += u[-1] < 0
        k = len(d)
        while k > 0 and d[k-1] <= 0.5:
            k -= 1
        hold += k <= len(d) - 3
    return inv, hold
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=110)
for j, seed in enumerate((351, 352, 353)):
    ax = axes[0, j]
    for arm in ('ctrl', 'k12'):
        b = buckets(bucket_path(arm, seed))
        ax.plot([x for x, _ in b], [y for _, y in b], color=COL[arm], lw=2, marker='o', ms=4, label=NAME[arm])
    ax.axvspan(30, 40, color='#eb6834', alpha=0.08)
    ax.set_title(f'training seed {seed}: train reach, 5M buckets', fontsize=10)
    ax.set_xlabel('training ticks (M)'); ax.set_ylim(0, 0.2)
    if j == 0:
        ax.set_ylabel('reach (episode-weighted)'); ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax = axes[1, j]
    vals = {arm: sweep(arm, seed) for arm in ('ctrl', 'k12')}
    xs = [0, 1]
    for i, arm in enumerate(('ctrl', 'k12')):
        ax.bar([x + (i - 0.5) * 0.36 for x in xs], vals[arm], width=0.34, color=COL[arm])
        for x, v in zip(xs, vals[arm]):
            ax.text(x + (i - 0.5) * 0.36, v + 0.3, f'{v}/20', ha='center', fontsize=9)
    ax.hlines(1, -0.45, 0.45, color='#eb6834', lw=1, ls='--')
    ax.set_xticks(xs, ['inverted at 20 s', 'close-and-hold'])
    ax.set_ylim(0, 12); ax.set_title(f'mean policy at 40M, eval seeds 351–370', fontsize=10)
    if j == 0:
        ax.set_ylabel('seeds of 20')
for a in axes.flat:
    a.spines[['top', 'right']].set_visible(False); a.grid(alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig('upright-shaping-replicate.png')

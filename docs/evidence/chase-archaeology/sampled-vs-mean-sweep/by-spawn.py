import re
pat = re.compile(r'distance_m=([\d.]+)')
def load(path):
    return [float(m[1]) for m in map(pat.search, open(path)) if m]
rows = []
for t in (351, 352, 353):
    for s in range(351, 371):
        for arm in ('ctrl', 'floor07'):
            rows.append(('mean', load(f'../entropy-floor-ab/s{t}/traces/{arm}-{s}.txt')))
            rows.append(('sampled', load(f'traces/{arm}-s{t}-sampled-{s}.txt')))
print('mode     spawn distance  n   ever<=0.5m  median closed in 20 s')
for mode in ('mean', 'sampled'):
    for lo, hi in ((0, 2.5), (2.5, 4.5), (4.5, 10)):
        r = [d for m, d in rows if m == mode and lo <= d[0] < hi]
        cl = sorted(d[0] - d[-1] for d in r)
        print(f'{mode:8} {lo:>3}-{hi:<4} m      {len(r):3} {sum(min(d) <= 0.5 for d in r):5}       {cl[len(cl) // 2]:.2f} m')

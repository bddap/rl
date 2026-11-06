import re
pat = re.compile(r'distance_m=([\d.]+)')
def load(path):
    return [float(m[1]) for m in map(pat.search, open(path)) if m]
print('arm  spawn distance  n   ever<=0.5m  median closed in 20 s')
for arm in ('off', 'on'):
    rows = [load(f'traces/{arm}-s{t}-{s}.txt') for t in (351, 352, 353) for s in range(351, 371)]
    for lo, hi in ((0, 2.5), (2.5, 4.5), (4.5, 10)):
        r = [d for d in rows if lo <= d[0] < hi]
        cl = sorted(d[0] - d[-1] for d in r)
        med = f'{cl[len(cl) // 2]:.2f} m' if cl else '-'
        print(f'{arm:4} {lo:>3}-{hi:<4} m      {len(r):3} {sum(min(d) <= 0.5 for d in r):5}       {med}')

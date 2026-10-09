import re, sys, glob, os
pat = re.compile(r'ROLLOUT_TRACE video_tick=(\d+) .*?distance_m=([\d.]+) tip_distance_m=(?:Some\(([\d.e-]+)\)|None) upright=(-?[\d.]+)')
def load(path):
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append((int(m[1]) // 64, float(m[2]), float(m[3]) if m[3] else float('inf'), float(m[4])))
    return rows
def hold(ds):
    n = len(ds)
    k = n
    while k > 0 and ds[k-1] <= 0.5:
        k -= 1
    return k <= n - 3, (k + 1 if k < n else None)
out = []
for c in ('40000512', '48172032'):
    stats = dict(n=0, inv20=0, ever=0, righted=0, hold=0, ever05=0, tiphold=0, fail_inv=0, fail_up=0, closed=[])
    lines = []
    for s in range(351, 371):
        r = load(f'traces/{c}-{s}.txt')
        assert len(r) == 20, (c, s, len(r))
        t, d, tip, up = zip(*r)
        inv = [u < 0 for u in up]
        inv20 = inv[-1]
        ever = any(inv)
        first_inv = inv.index(True) if ever else None
        righted = ever and any(u >= 0.5 for u in up[first_inv+1:])
        h, hstart = hold(d)
        th, _ = hold(tip)
        e05 = min(d) <= 0.5
        stats['n'] += 1; stats['inv20'] += inv20; stats['ever'] += ever; stats['righted'] += righted
        stats['hold'] += h; stats['ever05'] += e05; stats['tiphold'] += th
        if not h:
            stats['fail_inv' if inv20 else 'fail_up'] += 1
        closed = d[0] - d[-1]
        stats['closed'].append(closed)
        tinv = f'{first_inv+1}s' if ever else '-'
        lines.append(f'  {s}  d {d[0]:5.2f}->{min(d):5.2f}(min)->{d[-1]:5.2f}  tip_min {min(tip):5.2f}  upright first {up[0]:+.2f} min {min(up):+.2f} end {up[-1]:+.2f}  first_inv {tinv:>4}  inv20 {int(inv20)} righted {int(righted)} hold {int(h)}' + (f'@{hstart}s' if h else ''))
    st = stats
    cl = sorted(st['closed'])
    out.append(f'checkpoint {c}:')
    out += lines
    out.append(f'  SUMMARY {c}: inverted@20s {st["inv20"]}/20  ever-inverted {st["ever"]}/20  righted {st["righted"]}/20 ({st["righted"]}/{st["ever"]} of ever-inverted)  close-and-hold(body<=0.5m) {st["hold"]}/20  ever<=0.5m {st["ever05"]}/20  tip-hold {st["tiphold"]}/20  failures: inverted {st["fail_inv"]} upright {st["fail_up"]}  median distance closed {cl[9]/2+cl[10]/2:.2f} m  mean {sum(cl)/20:.2f} m')
print('\n'.join(out))

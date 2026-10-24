import re, glob
from math import comb
line = re.compile(r'^  (\d{3})  d .*? hold (\d)')
def holds(path):
    out, arm = {}, None
    for l in open(path):
        if l.startswith('checkpoint '): arm = l.split()[1].rstrip(':')
        m = line.match(l)
        if m: out[(arm, int(m[1]))] = int(m[2])
    return out
tot = {'mean': 0, 'sampled': 0}; b = c = 0
for t in (351, 352, 353):
    h = {m: holds(f'scorer-{m}-s{t}.txt') for m in ('mean', 'sampled')}
    for arm, name in (('ctrl', 'control -1'), ('k12', 'floor -0.7')):
        mn = sum(v for (a, _), v in h['mean'].items() if a == arm)
        sp = sum(v for (a, _), v in h['sampled'].items() if a == arm)
        print(f'training seed {t} {name}: close-and-hold mean {mn}/20 sampled {sp}/20')
    for k in h['mean']:
        tot['mean'] += h['mean'][k]; tot['sampled'] += h['sampled'][k]
        b += h['sampled'][k] and not h['mean'][k]; c += h['mean'][k] and not h['sampled'][k]
n = b + c
p = sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0
print(f'pooled close-and-hold: mean {tot["mean"]}/120 sampled {tot["sampled"]}/120')
print(f'discordant pairs: sampled-only b={b} mean-only c={c}; one-sided exact McNemar (H1 b>c) p={p:.4g}')
gap = tot['sampled'] >= 2 * tot['mean'] and p < 0.05
print('verdict:', 'DETERMINISTIC-POLICY GAP' if gap else 'NOT LEARNED TO CLOSE')

import re, math, sys
pat_t = re.compile(r'\((\d+) ticks\) \| reward'); pat_r = re.compile(r'reach ([0-9.]+) over (\d+) ep')
def window(path, lo=30_000_000, hi=40_000_000):
    s = n = 0.0
    for line in open(path):
        if '[learner] iter ' not in line: continue
        t = pat_t.search(line); r = pat_r.search(line)
        if not (t and r): continue
        tk = int(t[1])
        if lo <= tk < hi:
            s += float(r[1]) * int(r[2]); n += int(r[2])
    return s, n
tot = {}
for arm in ('ctrl', 'k12'):
    S = N = 0
    for seed in (351, 352, 353):
        s, n = window(f'{sys.argv[1]}/{arm}-s{seed}.log')
        print(f'{arm} seed {seed}: 30-40M reach {s/n:.4f} over {int(n)} episodes (reached {s:.0f})')
        S += s; N += n
    tot[arm] = (S, N)
    print(f'{arm} pooled 351-353: {S/N:.4f} over {int(N)} episodes')
(sc, nc), (sk, nk) = tot['ctrl'], tot['k12']
pc, pk = sc/nc, sk/nk; p = (sc+sk)/(nc+nk)
z = (pc - pk) / math.sqrt(p*(1-p)*(1/nc + 1/nk))
pval = 0.5 * math.erfc(z / math.sqrt(2))
print(f'one-sided H1 control > k12: z = {z:.3f}, p = {pval:.4g}; k12 below control at p<0.05: {pval < 0.05}')

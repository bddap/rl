import re, math, sys
from math import comb
holds = {}
for arm, label in (('off', 'ctrl'), ('on', 'k12')):
    holds[arm] = [int(re.search(rf'SUMMARY {label}: .*?close-and-hold\(body<=0.5m\) (\d+)/20', open(f'scorer-s{t}.txt').read())[1]) for t in (351, 352, 353)]
c, x = sum(holds['off']), sum(holds['on'])
k = c + x
p = sum(comb(60, j) * comb(60, k - j) for j in range(x, k + 1)) / comb(120, k) if k else 1.0
with open('pooled-hold.txt', 'w') as f:
    f.write(f'Control (gait clock off) holds by training seed 351-353: {holds["off"]}; pooled {c}/60\n')
    f.write(f'Treatment (gait clock on) holds by training seed 351-353: {holds["on"]}; pooled {x}/60\n')
    f.write(f'One-sided Fisher exact, H1 treatment > control, conditioning on {k} total holds: p = {p:.4f}\n')
    f.write(f'Closing gain at p<0.05: {p < 0.05}\n')
reach = {}
lines = []
for arm in ('off', 'on'):
    R = N = 0
    for t in (351, 352, 353):
        r = n = 0.0
        for line in open(f'buckets-{arm}-s{t}.raw'):
            m = re.match(r'(\d+) ([\d.]+) (\d+)', line)
            tick, rr, ep = int(m[1]), float(m[2]), int(m[3])
            if 30_000_000 <= tick < 40_000_000:
                r += rr * ep; n += ep
        lines.append(f'{arm} seed {t}: 30-40M reach {r / n:.4f} over {int(n)} episodes (reached {round(r)})')
        R += r; N += n
    reach[arm] = (R, N)
    lines.append(f'{arm} pooled 351-353: {R / N:.4f} over {int(N)} episodes')
(rc, nc), (rt, nt) = reach['off'], reach['on']
pc, pt = rc / nc, rt / nt
pp = (rc + rt) / (nc + nt)
z = (pc - pt) / math.sqrt(pp * (1 - pp) * (1 / nc + 1 / nt))
pz = 0.5 * math.erfc(z / math.sqrt(2))
lines.append(f'one-sided H1 control > treatment: z = {z:.3f}, p = {pz:.4g}; treatment below control at p<0.05: {pz < 0.05}')
open('pooled-reach.txt', 'w').write('\n'.join(lines) + '\n')
print(open('pooled-hold.txt').read() + '\n'.join(lines))

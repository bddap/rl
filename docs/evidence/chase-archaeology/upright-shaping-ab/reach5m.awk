# episode-weighted reach per 5M-tick bucket from [learner] iter lines
/\[learner\] iter / {
  if (!match($0, /\(([0-9]+) ticks\) \| reward/, t)) next
  if (!match($0, /reach ([0-9.]+) over ([0-9]+) ep/, r)) next
  b = int(t[1] / 5000000); s[b] += r[1] * r[2]; n[b] += r[2]; last = t[1]
  if (b > maxb) maxb = b
}
END { for (b = 0; b <= maxb; b++) if (n[b]) printf "%2d-%2dM  reach %.4f  episodes %d\n", b*5, b*5+5, s[b]/n[b], n[b]; printf "last tick %d\n", last }

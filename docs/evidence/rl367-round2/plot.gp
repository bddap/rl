# gnuplot -c plot.gp <before/trace.csv> <after/trace.csv> <out.png>
# Local-player altitude per sim tick from the RL_POS_TRACE `T` records, in player
# heights, time from the capture's settle tick (90).
set terminal pngcairo size 1100,420 font "DejaVu Sans,11" background "#ffffff"
set output ARG3
set datafile separator ","
h = 5096.0
set xlabel "time (s), 30 Hz ticks"
set ylabel "height (player heights)"
set grid lc rgb "#dddddd"
set key top left
set xrange [0:5.2]
set yrange [0:5.5]
plot '< grep "^T," '.ARG1 using (($2-90)/30.0):($5/h) with lines lw 2 lc rgb "#8a8a8a" title "before: 1.45 m/s liftoff, half-g float while rising", \
     '< grep "^T," '.ARG2 using (($2-90)/30.0):($5/h) with lines lw 2 lc rgb "#d1495b" title "after: 5 heights in 0.33 s, hang past the apex, shared-g fall"

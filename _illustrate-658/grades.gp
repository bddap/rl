set terminal pngcairo size 760,420 font "Helvetica,12"
set output "_illustrate-658/grades.png"
set title "fit.rs code-grade: final distribution at 81af11a (Work driven to 0)" font "Helvetica,13"
set style fill solid 1.0 border -1
set boxwidth 0.6
set yrange [0:20]
set xrange [-0.6:3.6]
set ylabel "criteria"
set grid ytics lc rgb "#cccccc"
unset key
set xtics ("Perfect" 0, "Good" 1, "Work" 2, "NA" 3) nomirror
# x value color(int)
$d << EOD
0 18 2592838
1 4 8369022
2 0 12606522
3 2 9803178
EOD
set label "18" at 0,18.7 center font "Helvetica,11"
set label "4"  at 1,4.7  center font "Helvetica,11"
set label "0  (was 1 -> fixed)" at 2,0.9 center font "Helvetica,10" tc rgb "#c0392b"
set label "2" at 3,2.7 center font "Helvetica,11"
plot $d using 1:2:3 with boxes lc rgb variable

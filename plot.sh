#!/usr/bin/env gnuplot

file=system("echo $A")
set term png size 1200,800
set output "imgs/".file.".png"
set view map
# set cbrange[0:0.06]
set xrange[0:600]
set yrange[0:400]
set palette defined (0 "black",12 "cyan", 16 "white");
plot "data/".file.".dat" binary array=600x400 format="%f" with image

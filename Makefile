PROGNAME:=lbm

CC:=OMPI_CC=gcc-14 mpicc

CFLAGS+= -std=c11 -Wall -Wextra -pedantic -Werror -fopenmp -O2
LDLIBS+= -lm

PROCS=1
THREADS=8

TARGETS= d2q7 d2q7_vec
IMAGES=$(shell ls data/*.dat 2> /dev/null| sed s/data/imgs/g | sed s/\.dat/.png/g)

all: $(TARGETS)

d2q7: src/d2q7.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o d2q7

d2q7_vec: src/d2q7_vec.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o d2q7_vec

run:
	OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) ./d2q7_vec

setup:
	mkdir data
	mkdir imgs

images: ${IMAGES}

imgs/%.png: data/%.dat
	echo "set term png size 1200,800; set output \"imgs/$*.png\"; set view 0,0,1; set cbrange [0:0.6]; set xrange[0:600]; set yrange[0:400]; set palette defined (0 \"black\",12 \"cyan\", 16\"white\"); plot \"data/$*.dat\" binary array=600x400 format=\"%f\" with image" | gnuplot -

anim: images
	ffmpeg -y -an -i imgs/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 vortex_shedding.mp4

clean:
	-rm data/*.dat imgs/*.png

purge:
	-rm data/*.dat imgs/*.png d2q7 vortex_shedding.mp4 compare

precision: src/precision.c
	gcc $^ -Wall -Wextra -pedantic -o compare

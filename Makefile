PROGNAME:=lbm

CC:=OMPI_CC=gcc-14 mpicc

CFLAGS+= -std=c11 -Wall -Wextra -pedantic -Werror -fopenmp -O2
LDLIBS+= -lm

PROCS=1
THREADS=8

TARGETS= d2q7 d2q7_vec d2q7_opt
IMAGES=$(shell ls data/*.dat 2> /dev/null| sed s/data/imgs/g | sed s/\.dat/.png/g)

all: $(TARGETS)

d2q7: src/d2q7.c src/domain.h
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o d2q7

d2q7_vec: src/d2q7_vec.c src/domain.h
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o d2q7_vec

d2q7_opt: src/d2q7_opt.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -o d2q7_opt

run:
	OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) ./d2q7

setup:
	mkdir data
	mkdir imgs

images: ${IMAGES}

imgs/%.png: data/%.dat
	echo "set term png size 1200,800; set output \"imgs/$*.png\"; set logscale zcb; set view 0,0,1; set cbrange[0.00000001:0.1]; set xrange[0:2400]; set yrange[0:1800]; splot \"data/$*.dat\" binary array=2400x1800 format=\"%f\" with pm3d" | gnuplot -

anim: images
	ffmpeg -y -an -i imgs/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 vortex_shedding.mp4

clean:
	-rm data/*.dat imgs/*.png

purge:
	-rm data/*.dat imgs/*.png d2q7 vortex_shedding.mp4 compare

precision: src/precision.c
	gcc $^ -Wall -Wextra -pedantic -o compare

CC:=OMPI_CC=gcc-14 mpicc

CFLAGS+= -std=c11 -Wall -Wextra -pedantic -Werror -O2
LDLIBS+= -lm

PROCS=1
THREADS=8

IMAGES=$(shell ls data/*.dat | sed s/data/imgs/g | sed s/\.dat/.png/g)

build: d2q7.c
	$(CC) $^ $(CFLAGS) $(LDLIBS) -fopenmp -o d2q7

run: d2q7
	OMP_NUM_THREADS=$(THREADS) mpirun -np $(PROCS) d2q7 -i 10000

images: ${IMAGES}

imgs/%.png: data/%.dat
	echo "set term png size 1200,800; set output \"imgs/$*.png\"; set view 0,0,1; set cbrange [0:0.6]; set palette defined (0 \"black\",12 \"cyan\", 16\"white\"); plot \"data/$*.dat\" binary array=600x400 format=\"%f\"" | gnuplot -

anim: images
	ffmpeg -y -an -i imgs/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 vortex_shedding.mp4

plot:
	ls data | parallel -v A={.} ./plot.sh

clean:
	-rm data/*.dat imgs/*.png

precision:
	gcc precision.c ${CFLAGS} -o compare

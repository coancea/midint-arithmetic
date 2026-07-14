CUB=cub-1.8.0

all: compile run

compile: main.cu helper.cu.h radix-sort-kers.cu.h radix-sort-kers-clean.cu.h
	nvcc -o our-sort main.cu

run:
	./our-sort 100000000	

clean:
	rm -f our_sort


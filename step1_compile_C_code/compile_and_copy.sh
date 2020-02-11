#!/bin/bash
gcc -Wall -pedantic -Wextra -std=c99 -msse2 -O3 -ftree-vectorize -ffast-math -funroll-loops -fomit-frame-pointer -m64 tvbii_multicore.c -o tvb -lgsl -lgslcblas -lm -lpthread 2> /output/build_output.txt
cp tvb /output/tvb
rm tvb tvbii_multicore.c



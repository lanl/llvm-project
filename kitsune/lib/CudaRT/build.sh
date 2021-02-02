#!/bin/bash

clang++ -c cudart.cpp -o cudart.o -I/home/ejpark/research2020/bolgrebolgre/kitsune/kitsune/include -I/usr/local/cuda-11.0/include
cp cudart.o ~/research2020/bolgrebolgre/EJ/

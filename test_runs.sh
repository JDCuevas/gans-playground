#!/bin/bash

for bs in 16 32 64 128
do
    for ci in 4 5 6 7
    do
        for nd in 2 3 4 5
        do
            for la in 0.01 0.1 1 10
            do
            python train.py --epochs 50 -bs $bs -ci $ci -nd $nd -la $la
            done
        done
    done
done

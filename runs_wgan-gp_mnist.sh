#!/bin/bash

for bs in 32 64 128 256
do
    for ci in 3 5 7 9
    do
        for nd in 32 64 128 256
        do
            for la in 1 5 10 15 
            do
            python train.py -m dcgan_2d -d mnist -l wgan_gp -e 100 -bs $bs -ci $ci -nd $nd -la $la -ne 16
            python train.py -m dcgan_2d -d mnist -l gan -e 100 -bs $bs -ci $ci -nd $nd -la $la -ne 16
            done
        done
    done
done
#!/bin/bash

mkdir data
mkdir data/measured
mkdir data/analysis
mkdir data/spice
mkdir data/cache
mkdir data/tmp
mkdir data/weights
mkdir data/devparams
mkdir data/datasets

cp $HOME/ini/experiments/adnn/*.sh .
cp $HOME/ini/experiments/adnn/*.py .
cp $HOME/ini/experiments/adnn/data/weights/weights_clipped.npy data/weights/
cp -r $HOME/ini/experiments/adnn/lib .
cp $HOME/ini/experiments/adnn/hyperopt.log .

date > INFO
vi INFO

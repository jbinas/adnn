#!/bin/bash

rm data/analysis/pwr.dat
rm data/analysis/pwr.tmp

for f in $(ls ./data/analysis/data*)
do
    echo "processing $f"
    python extract_pwr.py $f >> data/analysis/pwr.tmp
done

python extract_pwr.py combine data/analysis/pwr.tmp

rm data/analysis/pwr.tmp

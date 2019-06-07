#!/bin/bash

echo "rndseed	din	dout	inp	tmax	nops	en	class	pat1(id)	pat2(id)	out	meta" > data/analysis/out.dat

for f in data/analysis/data-*
do
	echo "processing $f"
	python analysis.py noplot noprint $f >> data/analysis/out.dat
done

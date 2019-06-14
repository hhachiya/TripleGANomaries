#!/bin/bash

for zDim in 1000 200; do
	for ((tri=0; tri<2; tri++)); do
		for ((char=0; char<10; char++)); do
			for stopTrainThre in 0.05; do
				for noiseSigma in 0.155; do
					#python adversarialClassifier.py 0 $char $tri $noiseSigma $zDim 10000 $stopTrainThre 1 
					python adversarialClassifier.py 2 $char $tri $noiseSigma $zDim 10000 $stopTrainThre 1.0 0
				done

			done
		done
	done
done
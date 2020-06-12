#!/bin/bash

for ((tri=0; tri<2; tri++)); do
	for zDim in 256 128 64; do
		for stopTrainThre in 0.01 0.05; do
			for alpha in 0.1 0.5; do
				for ((char=0; char<10; char++)); do
					for beta in 0.5 1.0 2.0; do
						python adversarialClassifier.py 2 $char $tri 0 $zDim 5000 $stopTrainThre $beta $alpha					
					done
					
					python adversarialClassifier.py 0 $char $tri 0.155 $zDim 5000 $stopTrainThre 1 $alpha					
				done
			done
		done
	done
done

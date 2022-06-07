#!/bin/bash
N=1
for ((i=1;i<=N;i++));
do
	cat languages.txt | while read line 
	do
   
			echo $line
			echo $i
			# qsub -v lang=$line run-mbert.sh
			python3 preprocess.py train_monoling_${line} &
                        # python3 eval_heads.py train_xlmr_${line}
	done
done

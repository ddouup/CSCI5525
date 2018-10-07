# CSCI5525
### Packages used
* os, sys, time, math
* argparse
* numpy
* matplotlib.pyplot
* scipy.sparse.linalg
### Datasets
* boston.csv: 506 x 13, 2 classes (After modification of target)
* digits.csv: 1797 x 64, 10 classses
### Run the command for problem 3:
```
python3 LDA1dProjection.py
	-f [/PATH/TO/FILE]
	-n [NUMBER OF SPLIT]

python3 LDA2dGaussGM.py
	-f [/PATH/TO/FILE]
	-n [NUMBER OF SPLIT]
```

### Run the command for problem 4:
```
python3 logisticRegression.py
	-f [/PATH/TO/FILE]
	-n [NUMBER OF SPLIT]
	-t [TRAINING PERCENT] (eg. 10 25 50 75 100)
  
python3 naiveBayesGauss.py
	-f [/PATH/TO/FILE]
	-n [NUMBER OF SPLIT]
	-t [TRAINING PERCENT] (eg. 10 25 50 75 100)
```
The logisticRegression.py and naiveBayesGauss.py save the test error results to csv files of the same directory.<br><br>
Run
```
python3 plot.py
```
to generate graph.

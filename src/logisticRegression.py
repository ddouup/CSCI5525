import numpy as np
import sys, os, math, matplotlib
import argparse
from helper import randomSplit

def buildParser():
	parser = argparse.ArgumentParser(
		prog = __file__
	)

	parser.add_argument(
		"-f", "--filename",
		type = str,
		required = True,
		dest = "filename",
		help = "Path to data file"
	)

	parser.add_argument(
		"-n", "--num_splits",
		type = int,
		required = True,
		dest = "num_splits",
		help = "Number of splits"
	)

	parser.add_argument(
		"-t", "--train_percent",
		type = int,
		nargs='+',
		required = True,
		dest = "train_percent",
		help = "Train percent of each split"
	)

	parser.add_argument(
		"-v", "--verbose",
		action = "store_true",
		dest = "verbose",
		help = "Enable printing debug information"
	)

	return parser


class logisticRegression():	
	def __init__(self, alpha):
		self.alpha = alpha

	def sigmoid(x):
  		return 1 / (1 + math.exp(-x))
	
	def train(self, X, y):
		pass

	def predict(self, X, y):

		return error



def main():
	parser = buildParser()
	args = parser.parse_args()

	if not os.path.isfile(args.filename):
		sys.exit(
			"ERROR: File does not exist"
		)

	data = np.genfromtxt(args.filename, delimiter=',')

	X = data[:,:-1]
	y = data[:,-1]

	
	if os.path.basename(args.filename) == 'boston.csv':
		print('Modify the target of boston dataset...')
		print()

		median = np.median(y)
		y = (y < median).astype(int)

	
	rs = randomSplit(args.num_splits,args.train_percent)
	for train_index, test_index in rs.split(y):

		X_train = X[train_index]
		y_train = X[train_index]
		X_test = y[test_index]
		y_test = y[test_index]

		model = logisticRegression(1)
		model.train(X_train, y_train)
		error = model.predict(X_test, y_test)


if __name__ == '__main__':
	main()
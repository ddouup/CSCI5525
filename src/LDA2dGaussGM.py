import numpy as np
import sys, os, math, matplotlib
import argparse
from helper import KFold

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
		"-n", "--num_crossval",
		type = int,
		required = True,
		dest = "num_crossval",
		help = "Number of splits"
	)

	parser.add_argument(
		"-v", "--verbose",
		action = "store_true",
		dest = "verbose",
		help = "Enable printing debug information"
	)

	return parser

class LDA2dGaussGM():	
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

	
	kfold = KFold(args.num_crossval)
	for train_index, test_index in kfold.split(y):

		X_train = X[train_index]
		y_train = X[train_index]
		X_test = y[test_index]
		y_test = y[test_index]

		model = LDA2dGaussGM(1)
		model.train(X_train, y_train)
		error = model.predict(X_test, y_test)


if __name__ == '__main__':
	main()
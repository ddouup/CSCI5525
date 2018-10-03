import numpy as np
import matplotlib.pyplot as plt
import sys, os, math
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

class LDA1dProjection():	
	def __init__(self, X_train, y_train):
		self.X = X_train
		self.X1 = X_train[np.where(y_train==0)]
		self.X2 = X_train[np.where(y_train==1)]
		self.y = y_train
		self.num = X_train.shape[1] # number of features


	def train(self):
		m1 = np.mean(self.X1, axis=0).reshape(self.num, 1)
		m2 = np.mean(self.X2, axis=0).reshape(self.num, 1)
		Sw = self.covariance(self.X1, m1, self.X2, m2)
		self.w = np.dot(np.linalg.inv(Sw), (m2-m1))
		result = np.dot(self.X, self.w)
		return result


	def predict(self, X_test):
		result = np.dot(X_test, self.w)
		return result


	def covariance(self, X1, m1, X2, m2):
		Sw = np.zeros((self.num, self.num))
		for i in range(X1.shape[0]):
			row = X1[i].reshape(self.num,1)
			Sw += np.dot((row-m1), (row-m1).T)

		for j in range(X2.shape[0]):
			row = X2[j].reshape(self.num,1)
			Sw += np.dot((row-m2), (row-m2).T)

		return Sw



def projection_plt(train_result, y_train, test_result, y_test):
	bins = 20

	plt.subplot(1,2,1)
	plt.hist(train_result[np.where(y_train==0)], bins=bins, alpha=0.5)
	plt.hist(train_result[np.where(y_train==1)], bins=bins, alpha=0.5)
	plt.title("Training data")

	plt.subplot(1,2,2)
	plt.hist(test_result[np.where(y_test==0)], bins=bins, alpha=0.5)
	plt.hist(test_result[np.where(y_test==1)], bins=bins, alpha=0.5)
	plt.title("Test data")
	plt.show()



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
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = LDA1dProjection(X_train, y_train)

		train_result = model.train()
		test_result = model.predict(X_test)

		projection_plt(train_result, y_train, test_result, y_test)


if __name__ == '__main__':
	main()
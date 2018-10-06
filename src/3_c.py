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

class LDA2dGaussGM():	
	def __init__(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)	# number of unique labels
		self.num = X_train.shape[0]			# number of instances
		self.feature_num = X_train.shape[1]			# number of featuress
		print("Class number: ", self.label_num)
		print("Training data size: ", self.num, 'x', self.feature_num)
		print()

	def train(self):
		means = self.cal_means()
		Sb, Sw = self.covariance(means)
		
		sys.exit()
		self.w = np.dot(np.linalg.inv(Sw), (m2-m1))
		result = np.dot(self.X, self.w)
		return result


	def predict(self, X_test):
		result = np.dot(X_test, self.w)
		return result


	def covariance(self, means):
		print("Calculating covariance...")
		print()

		Sb = np.zeros((self.feature_num, self.feature_num))
		Sw = np.zeros((self.feature_num, self.feature_num))

		overall_mean = np.mean(self.X, axis=0).reshape(self.feature_num,1)
		for i in range(self.label_num):
			n = self.X[self.y==self.labels[i]].shape[0]
			mean = means[i].reshape(self.feature_num,1)
			Sb += n*np.dot((mean-overall_mean), (mean-overall_mean).T)

		for i in range(self.label_num):
			mean = means[i].reshape(self.feature_num,1)
			for row in self.X[np.where(self.y==self.labels[i])]:
				Sw += np.dot((row-mean), (row-mean).T)

		return Sb, Sw


	def cal_means(self):
		means = np.zeros((self.label_num, self.feature_num))
		for i in range(self.label_num):
			temp = np.mean(self.X[np.where(self.y==self.labels[i])], axis=0).reshape(1, self.feature_num)
			means[i] = temp

		return means


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

	data = np.genfromtxt(args.filename, delimiter=',', dtype=int)

	X = data[:,:-1]
	y = data[:,-1]

	print('Data size:',X.shape)
	print()
	
	kfold = KFold(args.num_crossval)
	for train_index, test_index in kfold.split(y):

		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = LDA2dGaussGM(X_train, y_train)

		train_result = model.train()
		test_result = model.predict(X_test)

		projection_plt(train_result, y_train, test_result, y_test)

if __name__ == '__main__':
	main()
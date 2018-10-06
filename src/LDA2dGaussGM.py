import numpy as np
import matplotlib.pyplot as plt
import sys, os, math
import argparse
from helper import KFold
from scipy.sparse.linalg import eigs
from naiveBayesGauss import naiveBayesGauss

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
	def __init__(self, X_train, y_train, pro_dimension=1):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)			# number of unique labels
		self.num = X_train.shape[0]					# number of instances
		self.feature_num = X_train.shape[1]			# number of featuress
		self.pro_dimension = pro_dimension			# projection dimension

		print("Class number: ", self.label_num)
		print("Training data size: ", self.num, 'x', self.feature_num)
		print()

	def train(self):
		self.means = self.cal_means()
		Sb, Sw = self.covariance(self.means)
		eigvals, eigvecs = eigs(np.dot(np.linalg.pinv(Sw), Sb), k=self.pro_dimension, which='LM')

		self.w = eigvecs
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
				row = row.reshape(self.feature_num,1)
				Sw += np.dot((row-mean), (row-mean).T)

		return Sb, Sw


	def cal_means(self):
		means = np.zeros((self.label_num, self.feature_num))
		for i in range(self.label_num):
			temp = np.mean(self.X[np.where(self.y==self.labels[i])], axis=0).reshape(1, self.feature_num)
			means[i] = temp

		return means


def projection_plt(train_result, y_train, test_result, y_test, itr):
	bins = 20
	labels = np.unique(y_test)

	plt.subplot(5,4,2*itr-1)
	for i in range(len(labels)):
		result_k = train_result[np.where(y_train==i)]
		plt.scatter(result_k[:,0], result_k[:,1])
	plt.title("Training data of "+str(itr)+" split")

	plt.subplot(5,4,2*itr)
	for i in range(len(labels)):
		result_k = test_result[np.where(y_test==i)]
		plt.scatter(result_k[:,0], result_k[:,1])
	plt.title("Test data of "+str(itr)+" split")

	plt.draw()

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

	print('Data size:',X.shape)
	print()

	if os.path.basename(args.filename) == 'boston.csv':
		print('Modify the target of boston dataset...')
		print()

		median = np.median(y)
		y = (y < median).astype(int)


	error = np.ones((args.num_crossval,1))

	kfold = KFold(args.num_crossval)
	itr = 1
	for train_index, test_index in kfold.split(y):

		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = LDA2dGaussGM(X_train, y_train, pro_dimension=2)
		train_result = model.train()
		test_result = model.predict(X_test)

		projection_plt(train_result, y_train, test_result, y_test, itr)

		model = naiveBayesGauss().fit(train_result, y_train)
		error[itr-1] = model.score(test_result, y_test)

		itr += 1


	for i in range(args.num_crossval):
		print('Test error of '+str(i+1)+' split:', error[i])

	print('Test error mean:', np.mean(error))
	print('Test error std:', np.std(error))

	plt.show()


if __name__ == '__main__':
	main()
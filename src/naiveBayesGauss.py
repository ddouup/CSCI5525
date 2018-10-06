import numpy as np
import sys, os, math, argparse, time
from helper import randomSplit

from numpy import dot
from numpy.linalg import inv

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



class naiveBayesGauss():	
	def __init__(self):
		print()
	
	def fit(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features

		print("Class number: ", self.label_num)
		print("Training data size: ", self.X.shape)
		print()

		self.X=self.X+np.random.normal(0, 0.001, self.X.shape) #to prevent numerical problem

		self.prior = np.zeros(self.labels.shape)
		self.means = np.zeros((self.label_num, self.feature_num))
		self.stds = np.zeros((self.label_num, self.feature_num))
		for k in range(self.label_num):
			self.prior[k] = np.sum(self.y == self.labels[k])/self.y.shape[0]
			self.means[k] = np.mean(self.X[np.where(self.y == self.labels[k])], axis=0)
			self.stds[k] = np.std(self.X[np.where(self.y == self.labels[k])], axis=0, ddof=1)

		return self


	def cal_conditional(self, row, k):
		feature_num = self.feature_num

		Sigma = 1
		exp = 0
		for j in range(feature_num):
			Sigma *= self.stds[k][j]
			exp += (row[j] - self.means[k][j])**2/(2*self.stds[k][j]**2)

		divisor = (2*math.pi)**(feature_num/2)*Sigma
		conditional = np.exp(-exp)/divisor

		return conditional


	def predict(self, X_test):
		print("Test data size: ", X_test.shape)

		y_pre = np.array([], dtype=int)
		for i in range(X_test.shape[0]):
			row = X_test[i]

			conditional = np.zeros(self.labels.shape)
			a = np.zeros(self.labels.shape)
			evidence = 0
			for k in range(self.label_num):
				conditional[k] = self.cal_conditional(row, k)
				a[k] = np.log(self.prior[k])+np.log(conditional[k])
				evidence += np.exp(a[k])

			posterior = np.array([]) 
			for k in range(self.label_num):
				posterior = np.append(posterior, np.exp(a[k])/evidence)

			result = np.argmax(posterior)
			y_pre = np.append(y_pre, int(self.labels[result]))

		return y_pre

	def score(self, X_test, y_test):
		y_pre = self.predict(X_test)
		
		result = np.array([], dtype=int)
		num = X_test.shape[0]
		for i in range(num):
			result = np.append(result, y_pre[i] == y_test[i])

		error = 1 - np.sum(result)/num

		print("Test error rate:", error)
		print()

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

	print('Data size:', X.shape)
	print()

	start = time.time()

	error = np.ones((args.num_splits, len(args.train_percent)))

	rs = randomSplit(args.num_splits,args.train_percent)
	for train_index, test_index, i, p in rs.split(y):
		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = naiveBayesGauss().fit(X_train, y_train)
		error[i][p] = model.score(X_test, y_test)

	err_means = np.mean(error, axis=0)
	err_stds = np.std(error, axis=0)
	print('Test error mean:', err_means)
	print('Test error std:', err_stds)
	end = time.time()
	print('Time consumed:'+str(end-start)+'s')


	output = np.concatenate([err_means.reshape(1, len(args.train_percent)), err_stds.reshape(1, len(args.train_percent))])
	output_file = 'naiveBayesGauss_'+os.path.basename(args.filename)
	print('Store test result to: ',output_file)
	np.savetxt(output_file, output, delimiter=",")


if __name__ == '__main__':
	main()
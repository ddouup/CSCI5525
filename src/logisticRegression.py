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



class logisticRegression():	
	def __init__(self, max_iter=1000, delta=0.0001, tolerance=0.0001):
		self.max_iter = max_iter
		self.delta = delta
		self.tolerance = tolerance
	
	def fit(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features
		self.w = np.random.normal(0,0.001, (self.label_num, self.feature_num))

		print("Class number: ", self.label_num)
		print("Training data size: ", self.X.shape)
		print()

		self.X=self.X+np.random.normal(0, 0.001, self.X.shape) #to prevent numerical problem

		for k in range(self.label_num):

			# y_classi[i] = 1 if belongs to labels[k] else y_classi[i] = 0
			y_classi = np.zeros(self.y.shape, dtype = int)
			for i in range(len(self.y)):
				if self.y[i] == self.labels[k]:
					y_classi[i] = 1

			self.w[k] = self.irls(self.X, y_classi, self.max_iter, self.delta, self.tolerance)

		return self

	def irls(self, X, y, max_iter, delta, tolerance):
		delta = np.array(np.repeat(delta, self.num)).reshape(1,self.num)
		R = np.eye(self.num)
		z = inv(R).dot(y)
		w_i = dot(inv(X.T.dot(R).dot(X)),(X.T.dot(R).dot(z)))

		itr = 0
		while itr < max_iter:
			w_old = w_i

			delr =  abs(y - X.dot(w_old)).T
			r = 1.0/np.maximum( delta, delr )
			R = np.diag(r[0])
			
			#R = np.diag(X.dot(Wi)*(1 - X.dot(Wi)).reshape(1, self.num)[0])

			z = X.dot(w_old)-inv(R).dot(X.dot(w_old)-y)
			w_i = inv(X.T.dot(R).dot(X)).dot(X.T).dot(R).dot(z)

			if np.sum(abs(w_i-w_old)) < tolerance:
				break

			itr += 1

		return w_i

	def predict(self, X_test):
		print("Test data size: ", X_test.shape)
		y_pre = np.array([], dtype=int)
		for i in range(X_test.shape[0]):
			row = X_test[i]
			likelihood = np.array([])
			for k in range(self.label_num):
				likelihood = np.append(likelihood, self.w[k].T.dot(row))

			result = np.argmax(softmax(likelihood))
			y_pre = np.append(y_pre, int(self.labels[result]))
		return y_pre

	def score(self, X_test, y_test):
		y_pre = self.predict(X_test)

		result = np.array([], dtype=int)
		num = X_test.shape[0]
		for i in range(num):
			result = np.append(result, y_pre[i] == y_test[i])

		error = 1 - np.sum(result)/num

		print("Test error rate:",error)
		print()

		return error



def softmax(x):
	exp = np.exp(x-np.max(x))
	norms = np.sum(exp, axis=0)
	return exp / norms



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

		model = logisticRegression().fit(X_train, y_train)
		error[i][p] = model.score(X_test, y_test)

	err_means = np.mean(error, axis=0)
	err_stds = np.std(error, axis=0)
	print('Test error mean:', err_means)
	print('Test error std:', err_stds)
	end = time.time()
	print('Time consumed:'+str(end-start)+'s')

	output = np.concatenate([err_means.reshape(1, len(args.train_percent)), err_stds.reshape(1, len(args.train_percent))])
	output_file = 'logisticRegression_'+os.path.basename(args.filename)
	print('Store test result to: ',output_file)
	np.savetxt(output_file, output, delimiter=",")


if __name__ == '__main__':
	main()
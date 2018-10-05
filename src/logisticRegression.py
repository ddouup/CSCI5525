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
	def __init__(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features
		self.W = np.random.normal(0,0.001, (self.label_num, self.feature_num))

		print("Class number: ", self.label_num)
		print("Training data size: ", self.num, 'x', self.feature_num)
		#self._train()
	
	def fit(self, X_train, y_train):
		self.X = X_train
		self.y = y_train
		self.labels = np.unique(y_train)
		self.label_num = len(self.labels)		# number of unique labels
		self.num = X_train.shape[0]				# number of instances
		self.feature_num = X_train.shape[1]		# number of features
		self.w = np.random.normal(0,0.001, (self.label_num, self.feature_num))

		self.X=self.X+np.random.normal(0, 0.001, self.X.shape) #to prevent numerical problem

		for i in range(self.label_num):

			# y_classi[j] = 1 if belongs to labels[i] else y_classi[j] = 0
			y_classi = np.zeros(self.y.shape, dtype = int)
			for j in range(len(self.y)):
				if self.y[j] == self.labels[i]:
					y_classi[j] = 1

			self.w[i] = self.irls(self.X, y_classi)

		print(self.w.shape)
		return self

	def irls(self, X, y, max_iter=1000, delta=0.0001, tolerance=0.0001):
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

		return y_pre

	def score(self, X_test, y_test):
		y_pre = self.predict(X_test)

		return score



def sigmoid(x):
	return 1 / (1 + math.exp(-x))



def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms



def main():

	parser = buildParser()
	args = parser.parse_args()

	if not os.path.isfile(args.filename):
		sys.exit(
			"ERROR: File does not exist"
		)

	data = np.genfromtxt(args.filename, delimiter=',', dtype = int)

	X = data[:,:-1]
	y = data[:,-1]

	
	if os.path.basename(args.filename) == 'boston.csv':
		print('Modify the target of boston dataset...')
		print()

		median = np.median(y)
		y = (y < median).astype(int)

	print('Data size:', X.shape)
	print()
	
	rs = randomSplit(args.num_splits,args.train_percent)
	start = time.time()
	for train_index, test_index in rs.split(y):

		X_train = X[train_index]
		y_train = y[train_index]
		X_test = X[test_index]
		y_test = y[test_index]

		model = logisticRegression(X_train,y_train).fit(X_train, y_train)
		error = model.score(X_test, y_test)

	end = time.time()
	print('Time consumed:'+str(end-start))
if __name__ == '__main__':
	main()
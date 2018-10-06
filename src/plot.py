import matplotlib.pyplot as plt
import numpy as np

def main():
	lr_boston = np.genfromtxt('logisticRegression_boston.csv', delimiter=',')
	nb_boston = np.genfromtxt('naiveBayesGauss_boston.csv', delimiter=',')
	
	lr_digits = np.genfromtxt('logisticRegression_digits.csv', delimiter=',')
	nb_digits = np.genfromtxt('naiveBayesGauss_digits.csv', delimiter=',')

	x = [10, 25, 50, 75, 100]

	plt.subplot(2,1,1)
	plt.plot(x,lr_boston[0], 'ro-', label='Logistic Regression')
	plt.plot(x,nb_boston[0], 'bo-', label='Naive Bayes')
	plt.xlabel("Training percent(%)")
	plt.ylabel("Error mean")
	plt.title("Boston 50")
	plt.legend()
	plt.grid()

	plt.subplot(2,1,2)
	plt.plot(x,lr_digits[0], 'ro-', label='Logistic Regression')
	plt.plot(x,nb_digits[0], 'bo-', label='Naive Bayes')
	plt.xlabel("Training percent(%)")
	plt.ylabel("Error mean")
	plt.title("Digits")
	plt.legend()
	plt.grid()
	plt.show()

if __name__ == '__main__':
	main()
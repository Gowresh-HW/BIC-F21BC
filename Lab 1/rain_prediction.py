#!/usr/bin/env python

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#shape[0] : number of rows
#shape[1] : number of columns

class RainPredictor:
	def __init__(self):
		self.data = pd.read_csv("seattleWeather_1948-2017.csv")
		#print(self.data)
		self.data=self.data.dropna()
		self.data.info()

		#print("rows = {}".format(data.shape[0]))
		#print("columns = {}".format(data.shape[1]))
		pd.DataFrame(self.data.dtypes).rename(columns = {0:"dtype"})
		

		self.df = self.data.copy()
		self.df['TMAX'] = self.df['TMAX'].astype(float)
		self.df['TMIN'] = self.df['TMIN'].astype(float)
		self.df['class'] = self.df['RAIN'].apply(lambda x : 1 if x == True else 0)
		self.df.shape[1]
		self.X = self.df[['PRCP','TMAX','TMIN']].copy()

		#print(self.X.dtypes)
		self.y = self.df['class'].copy()
		#print self.X
		#print self.y

		#self.X = (self.X - np.min(self.X))/(np.max(self.X )) #Normalisation
		

		intercept = np.ones((self.X.shape[0], 1))
		self.X = np.concatenate((intercept, self.X), axis=1)
		
		self.alpha = 0.0001
		print("Data has been read and prepared..!!") 
		print(self.X.shape[0])
		print(self.X.shape[1])
		self.alpha = input("Please input the learning rate:  ")
		self.max_itr = input("Please input the Maximum Iterations:  ")
		self.algorithm()
		

	def algorithm(self):
		self.start_time = time.time()
		
		self.theta = np.zeros(self.X.shape[1])
		print(self.theta)
		theta_cost = []
		for i in range(self.max_itr):
			z = self.sigmoid(self.X, self.theta)
    			#print("h=", h)
    			#print("y=", y)
    			gradient = self.gradient_descent(self.X,z, self.y)
    			#print("gradient =", gradient)
    			self.theta = self.update_weight_loss(self.theta, self.alpha, gradient)
			theta_cost.append(self.theta)
			#print(self.theta)
		print("Training time (Log Reg using Gradient descent):" + str(time.time() - self.start_time) + " seconds")
		print("Learning rate: {}\nIteration: {}".format(self.alpha, self.max_itr))

		print(self.theta)
		result = self.sigmoid(self.X, self.theta)

		f = pd.DataFrame(np.around(result, decimals=6)).join(self.y)
		f['pred'] = f[0].apply(lambda x : True if x < 0.5 else False)
		print("Accuracy (Loss minimization):")
		print(f.loc[f['pred']==f['class']].shape[0] / float(f.shape[0]) * 100)
		plt.plot(np.arange(self.max_itr), theta_cost)
		plt.show()

		#For Confusion Matrix
		YActual = f['class'].tolist()
		YPredicted =  f['pred'].tolist()

		#print(YActual)
		#print(YPredicted)

		TP = 0
		TN = 0
		FP = 0
		FN = 0

		for l1,l2 in zip(YActual, YPredicted):
    			if (l1 == 1 and  l2 == 1):
        			TP = TP + 1
    			elif (l1 == 0 and l2 == 0):
        			TN = TN + 1
    			elif (l1 == 1 and l2 == 0):
        			FN = FN + 1
    			elif (l1 == 0 and l2 == 1):
        			FP = FP + 1

		print("Confusion Matrix: ")

		print("TP=", TP)
		print("TN=", TN)
		print("FP=", FP)
		print("FN=", FN)

		# Precision = TruePositives / (TruePositives + FalsePositives)
		# Recall = TruePositives / (TruePositives + FalseNegatives)
		temp = float(TP+FP)
		print(TP, temp, TP/temp)
		P = TP/float(TP + FP)
		R = TP/float(TP + FN)

		print("Precision = ", P)
		print("Recall = ", R)

	def sigmoid(self, X, weight):
    		z = np.dot(X, weight)
   		# print("Z dim =", z.shape[0])
   		# print("z =", z)
    		return 1 / (1 + np.exp(-z))

	def gradient_descent(self, X, z, y):
		#X.T transpose of X
		#y.shape[0] is sample size (N in the learning material), we divide by N to find avg (batch mode)
    		return np.dot(X.T, (z - y)) / y.shape[0]

	def update_weight_loss(self, weight, learning_rate, gradient):
    		return weight - learning_rate * gradient			

if __name__ == '__main__':
	r = RainPredictor()


import numpy as np
import pandas as pd
import math
import random


class TSonLR:

    def __init__(self, sigma, dim):
        self.sigma = sigma
        self.dim   = dim
        self.theta = np.array([0 for i in range(dim)])
        self.H_t   = 0
        self.G_t   = 0
        self.max_val   = -9999
        self.max_index = -9999

    def Hesse(self, data, T):
        e = math.e
        sum = np.eye(self.dim) / self.sigma
        for t in range(T):
            a = np.array(data.iloc[t])
            sum += (e ** (np.dot(self.theta.T, a))) * (np.dot(np.matrix(a).T, np.matrix(a))) / (1 + e ** (np.dot(self.theta.T, a))) ** 2
        return sum

    def Gradient(self, data, T, rewards):
        def X(a, reward):
            return a if reward else 0
        e = math.e
        sum = self.theta / self.sigma
        for t in range(T):
            a = np.array(data.iloc[t])
            sum += (e ** (np.dot(self.theta.T, a))) * (a) / (1 + e ** (np.dot(self.theta.T, a))) - X(a, rewards[t])
        return sum

    def update_theta(self, data, T, rewards):
        while True:
            self.H_t = self.Hesse(data, T)
            self.G_t = self.Gradient(data, T, rewards)
            theta_tmp = self.theta - np.dot(np.linalg.inv(self.H_t), self.G_t)
            theta_tmp = np.ravel(theta_tmp)
            if np.allclose(self.theta, theta_tmp):
                break
            self.theta = theta_tmp

    def predict(self, a):
        theta_nami = np.random.multivariate_normal(self.theta, np.linalg.inv(self.H_t))
        product = np.dot(a.T, theta_nami)
        if self.max_val < product:
            self.max_val = product
            self.max_index = i
        return self.max_index



data = pd.read_csv("data/data.txt", sep="\t").iloc[:,:]
sigma = 0.5 ** 2
dim = 5
theta = np.array([0 for i in range(dim)])
rewards = [0]

tslr = TSonLR(sigma=sigma, dim=dim)

for t in range(1, 144):
    a = data.iloc[:t, :5]
    y = data.iloc[:t, 10]
    print(a, y)

    a_actual = data.iloc[t, :5]
    y_actual = data.iloc[t, 10]

    tslr.update_theta(a, t, rewards)
    y_pred = tslr.predict(a_actual)

    if pred == actual:
        rewards.append(1)
    else:
        rewards.append(0)

    print(t, pred)

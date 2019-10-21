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

    def Hesse(self, data, t):
        e = math.e
        sum = 0
        for i in range(t+1):
            a = np.array(data.iloc[i])
            sum += (e ** (np.dot(self.theta.T, a))) * (np.dot(np.matrix(a).T, np.matrix(a))) / (1 + e ** (np.dot(self.theta.T, a))) ** 2
        return sum

    def Gradient(self, data, t):
        e = math.e
        sum = 0
        for i in range(t+1):
            a = np.array(data.iloc[i])
            sum += (e ** (np.dot(self.theta.T, a))) * (a) / (1 + e ** (np.dot(self.theta.T, a)))
        return sum

    def update_theta(self, data, t):
        while True:
            self.H_t = np.eye(self.dim) / self.sigma + self.Hesse(data, t)
            self.G_t = self.theta / self.sigma + self.Gradient(data, t)
            theta_tmp = self.theta - np.dot(np.linalg.inv(self.H_t), self.G_t)
            theta_tmp = np.ravel(theta_tmp)
            if np.allclose(self.theta, theta_tmp):
                break
            self.theta = theta_tmp

    def predict(self, a_t):
        theta_nami = np.random.multivariate_normal(self.theta, np.linalg.inv(self.H_t))
        for i in range(len(a_t)):
            product = np.dot(a_t[i].T, theta_nami)
            if self.max_val < product:
                self.max_val = product
                self.max_index = i
        return self.max_index



data = pd.read_csv("data/data.txt", sep="\t").iloc[:,:5]
sigma = 0.5 ** 2
dim = 5
theta = np.array([0 for i in range(dim)])

tslr = TSonLR(sigma=sigma, dim=dim)

for t in range(0, 144):
    a_t = [0, 0, 0]
    a_t[0] = np.array([random.uniform(0, 100) for i in range(dim)])
    a_t[1] = np.array([random.uniform(50, 150) for i in range(dim)])
    a_t[2] = np.array([random.uniform(100, 200) for i in range(dim)])

    tslr.update_theta(data, t)
    max_index = tslr.predict(a_t)

    print(t, max_index)

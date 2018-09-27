import numpy as np
import pandas as pd
import math
import random

def Hesse(theta, data, t):
    e = math.e
    sum = 0
    for i in range(t+1):
        a = np.array(data.iloc[i])
        sum += (e ** (np.dot(theta.T, a))) * (np.dot(np.matrix(a).T, np.matrix(a))) / (1 + e ** (np.dot(theta.T, a))) ** 2
    return sum

def Gradient(theta, data, t):
    e = math.e
    sum = 0
    for i in range(t+1):
        a = np.array(data.iloc[i])
        sum += (e ** (np.dot(theta.T, a))) * (a) / (1 + e ** (np.dot(theta.T, a)))
    return sum

# data = pd.read_csv("data/data.txt", sep="\t").iloc[:,:5]
sigma = 0.5 ** 2
d = 5
theta = np.array([0 for i in range(d)])

for t in range(0, 144):
    a_t = [0, 0, 0]
    a_t[0] = np.array([random.uniform(0, 100) for i in range(5)])
    a_t[1] = np.array([random.uniform(0, 100) for i in range(5)])
    a_t[2] = np.array([random.uniform(0, 100) for i in range(5)])

    while True:
        H_t = np.eye(d) / sigma + Hesse(theta, data, t)
        G_t = theta / sigma + Gradient(theta, data, t)
        theta_tmp = theta - np.dot(np.linalg.inv(H_t), G_t)
        theta_tmp = np.ravel(theta_tmp)
        if np.allclose(theta, theta_tmp):
            break
        theta = theta_tmp
        print (theta, theta_tmp)
    theta_nami = np.random.multivariate_normal(theta, np.linalg.inv(H_t))

    max_val, max_index = -9999 , -9999
    for i in range(len(a_t)):
        product = np.dot(a_t[i].T, theta_nami)
        if max_val < product:
            max_val = product
            max_index = i

    print(max_index)
    print("======================================================")

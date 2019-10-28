import collections
import numpy as np
import pandas as pd
import math


class TSonLR:
    def __init__(self, sigma, dim):
        self.sigma = sigma
        self.dim = dim
        self.theta = np.array([0 for i in range(dim)])
        self.H_t = np.eye(self.dim) / self.sigma
        self.G_t = self.theta / self.sigma
        self.a_hist = []

    def Hesse(self, data, T):
        e = math.e
        sum = np.eye(self.dim) / self.sigma
        self.a_hist.append(data)
        for t in range(T-1):
            a = np.array(self.a_hist[t])
            sum += (e ** (np.dot(self.theta.T, a))) * (np.dot(np.matrix(a).T,
                                                              np.matrix(a))) / (1 + e ** (np.dot(self.theta.T, a))) ** 2
        return sum

    def Gradient(self, data, T, rewards):
        def X(a, reward):
            return a if reward else 0
        e = math.e
        sum = self.theta / self.sigma
        for t in range(T-1):
            a = np.array(self.a_hist[t])
            sum += (e ** (np.dot(self.theta.T, a))) * (a) / \
                (1 + e ** (np.dot(self.theta.T, a))) - X(a, rewards[t])
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

    def predict(self, a_sets):
        # print(self.theta)
        # print((self.H_t))
        theta_nami = np.random.multivariate_normal(
            self.theta, np.linalg.inv(self.H_t))
        max_val = -9999
        max_index = -9999
        for i in range(len(a_sets)):
            product = np.dot(np.array(a_sets[i]).T, theta_nami)
            if max_val <= product:
                max_val = product
                max_index = i
        return max_index


class BehaviorDistribution:
    def __init__(self):
        self.behaviors = list(
            map(str.strip, open("dataset/behaviors.txt").readlines()))
        self.beta = 10 ** (0.0)  # 事前分布の重み
        self.prior_dist = pd.read_csv("dataset/behavior_distribution.csv")
        self.user_dist = pd.read_csv("dataset/user_distribution.csv")

    def inc_distribution(self, time, action):
        # 時刻のactionのカウントを +1
        row = self.user_dist[self.user_dist.name == action].index[0]
        self.user_dist.iloc[row, time+1] += 1
        return None

    def dec_distribution(self, time, action):
        # 時刻のactionのカウントを -1
        row = self.user_dist[self.user_dist.name == action].index[0]
        self.user_dist.iloc[row, time+1] -= 1
        return None

    # 時刻timeの全行為者率の取得
    def get_distribution(self, time):
        hour = math.floor(time / 60)
        minute = time - hour * 60
        index = hour * 60 + minute
        prior_df = self.prior_dist
        user_df = self.user_dist

        # 時刻tのときの行為率を全て取得
        try:
            col = index + 1
            behavior_rate = (user_df.iloc[:, col] + self.beta * prior_df.iloc[:, col]) / (
                sum(user_df.iloc[:, col]) + self.beta * sum(prior_df.iloc[:, col]))
        except:
            col = index + 14 + 1
            behavior_rate = (user_df.iloc[:, col] + self.beta * prior_df.iloc[:, col]) / (
                sum(user_df.iloc[:, col]) + self.beta * sum(prior_df.iloc[:, col]))

        return behavior_rate.values

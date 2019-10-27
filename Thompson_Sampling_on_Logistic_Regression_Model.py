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


def get_context(time, action, p, m, d):
    def update_param(action, p, m, d):
        def split_meal_behavior(action):
            if "cook" in action.split("_"):
                action = "cook"
            elif "eat" in action.split("_"):
                action = "eat"
            elif "wash" in action.split("_"):
                action = "wash"
            return action

        def zero_one(param):
            if param > 1:
                return 1
            elif param < 0:
                return 0
            else:
                return param

        action = split_meal_behavior(action)
        inc_p = LIFE_MODEL[LIFE_MODEL.behavior ==
                           action].physical_fatigue.values[0]
        inc_m = LIFE_MODEL[LIFE_MODEL.behavior ==
                           action].mental_fatigue.values[0]
        inc_d = LIFE_MODEL[LIFE_MODEL.behavior ==
                           action].discomfort_index.values[0]
        p = zero_one(inc_p + p)
        m = zero_one(inc_m + m)
        d = zero_one(inc_d + d)
        return p, m, d

    def update_act_rate(time):
        return list(beh.get_distribution(time))

    p, m, d = update_param(action, p, m, d)
    beh.inc_distribution(time, action)
    vector = update_act_rate(time)
    beh.dec_distribution(time, action)
    vector.extend([p, m, d])

    return vector


LIFE_MODEL = pd.read_csv("dataset/life_model.csv")
data = pd.read_csv("dataset/life_log.csv")
data.datetime = pd.to_datetime(data.datetime)
sigma = 1.0 ** 2
dim = 22  # 特徴量の次元数
p, m, d = 0.5, 0.5, 0.5
rewards = [0]

# インスタンス
tslr = TSonLR(sigma=sigma, dim=dim)
beh = BehaviorDistribution()

length = data.shape[0]
behaviors = beh.behaviors[1:-2]

for t in range(1, length):

    # 現在の行動 (パラメータ)に基づく反実仮想を含むコンテキストの取得
    record = data.iloc[t, :]
    time_min = record.datetime.hour * 60 + record.datetime.minute
    action_current = record.action
    a_actual = get_context(time_min, action_current, p, m, d)  # 特徴量

    if action_current in ["go_to_bed", "leave_home"]:
        print(2222, action_current)
        continue

    a_counterfactuals = []
    for behavior in behaviors:
        try:
            time_min_next = time_min + 15
            a_counterfactual = get_context(time_min_next, behavior, p, m, d)
        except:
            time_min_next = time_min + 14
            a_counterfactual = get_context(time_min_next, behavior, p, m, d)
        # print(time_min_next)

        printlist = ['{:.2f}'.format(val) for val in a_counterfactual]
        a_counterfactuals.append(a_counterfactual)
        # print(printlist, time_min_next, behavior)

    # 最適な次の行動を取得
    y_pred = tslr.predict(a_counterfactuals)

    record = data.iloc[t+1, :]
    action_next = record.action
    y_actual = behaviors.index(action_next)  # 実際に取った行動

    # 報酬の獲得
    reward = 1 if y_pred == y_actual else 0
    rewards.append(reward)
    print(reward)
    # パラメータの更新
    beh.inc_distribution(time_min, action_current)  # 行為者率の更新
    p, m, d = a_actual[-3:]  # パラメータの更新


# for t in range(1, length):
#     record = data.iloc[t, :]
#     time_min = record.datetime.hour * 60 + record.datetime.minute
#     action = record.action

#     a_actual = get_context(time_min, action, p, m, d)  # 特徴量
#     y_actual = behaviors.index(action)  # 実際に取った行動

#     if t != 1:
#         record_prev = data.iloc[t-1, :]
#         time_min_prev = record_prev.datetime.hour * 60 + record_prev.datetime.minute
#         action_prev = record_prev.action
#         a_actual_prev = get_context(time_min_prev, action_prev, p, m, d)
#         tslr.update_theta(a_actual_prev, t, rewards)

#     # ifの腕
#     a_counterfactuals = []
#     for behavior in behaviors:
#         a_counterfactual = get_context(time_min, behavior, p, m, d)
#         print(a_counterfactual)
#         a_counterfactuals.append(a_counterfactual)

#     if t > 40:
#         break
#     y_pred = tslr.predict(a_counterfactuals)

#     # 報酬の取得
#     reward = 1 if y_pred == y_actual else 0
#     rewards.append(reward)
#     print(t, reward, y_pred, y_actual)
#     print(behaviors[y_pred], behaviors[y_actual])
#     print()

#     # 更新
#     beh.inc_distribution(time_min, action)  # 行為者率の更新
#     p, m, d = a_actual[-3:]  # パラメータの更新

# print(collections.Counter(rewards))

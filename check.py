import pandas as pd
import numpy as np
from TS import BehaviorDistribution
from TS import TSonLR


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

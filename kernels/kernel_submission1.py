import numpy as np
from sklearn import linear_model as lm

from libraries import kagglegym

env = kagglegym.make()
observation = env.reset()

train = observation.train
mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)

cols_to_use = ['technical_20']

low_y_cut = np.percentile(train.y, 5)
high_y_cut = np.percentile(train.y, 95)

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

models_dict = {}
for col in cols_to_use:
    model = lm.LinearRegression()
    model.fit(np.array(train.loc[y_is_within_cut, col].values).reshape(-1, 1), train.loc[y_is_within_cut, 'y'])
    models_dict[col] = model

col = 'technical_20'
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1, 1)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break

print(info)

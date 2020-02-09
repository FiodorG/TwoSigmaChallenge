import numpy as np
from sklearn import linear_model as lm
from utilities import kagglegym

env = kagglegym.make()
observation = env.reset()

train = observation.train
train['y_lag'] = train['technical_20'] - train['technical_30']
train = train[train.y_lag != 0]
mean_values = train.median(axis=0)
train.fillna(mean_values, inplace=True)

cols_to_use = ['y_lag']

low_y_cut = np.percentile(train.y, 5)
high_y_cut = np.percentile(train.y, 95)

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

models_dict = {}
for col in cols_to_use:
    model = lm.BayesianRidge()
    model.fit(np.array(train.loc[y_is_within_cut, col].values).reshape(-1, 1), train.loc[y_is_within_cut, 'y'])
    models_dict[col] = model
    
ymedian_dict = dict(train.groupby(["id"])["y"].median())

def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y

col = 'y_lag'
model = models_dict[col]
while True:
    
    observation.features.fillna(mean_values, inplace=True)
    observation.features['y_lag'] = observation.features['technical_20'] - observation.features['technical_30']
    test_x = np.array(observation.features[col].values).reshape(-1, 1)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
    target = observation.target
    
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    observation, reward, done, info = env.step(target)
    if done:
        break

print(info)
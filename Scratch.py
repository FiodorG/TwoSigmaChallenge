from __future__ import print_function
import numpy as np
import pandas as pd
import Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


with pd.HDFStore("train.h5", "r") as data_file:
    dataset = data_file.get("train")

column_all = dataset.columns
columns_derived = [c for c in dataset.columns if 'derived' in c] + ['timestamp', 'id']
columns_fundamental = [c for c in dataset.columns if 'fundamental' in c] + ['timestamp', 'id']
columns_technical = [c for c in dataset.columns if 'technical' in c] + ['timestamp', 'id']

df_y = dataset[['id', 'timestamp', 'y']]
# df_derived = dataset[columns_derived].groupby('timestamp')
# df_fundamental = dataset[columns_fundamental].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
# df_technical = dataset[columns_technical].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()

# Visualization.plot_column_per_id_and_timestamp(dataset, 'technical_7')
# Visualization.plot_column(dataset, 'y', 800, True)
# Visualization.plot_pca(dataset, 'derived_3')
# Visualization.plot_stock_prices(dataset, 'fundamental_28', 200, False)
# Visualization.plot_all_columns_for_id(dataset, 500, columns_derived, columns_fundamental, columns_technical)
Visualization.linear_regression(dataset, 500, 'y', 20)

#plot_all_graphs(df_derived)
#yMean = df['y']['mean']
#corrColumns = [c for c in df_derived.columns if 'timestamp' not in c]
#derived = np.array([df_derived[c] for c in corrColumns])
#Corr = np.corrcoef(yMean, derived)
#sns.clustermap(pd.DataFrame(np.abs(Corr), columns=['---y--']+[c[0].split('_')[1] for c in corrColumns]))

print(columns_derived)
print(columns_fundamental)
print(columns_technical)
print()

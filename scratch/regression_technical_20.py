column = 'technical_20'
model = linear_model.LinearRegression(normalize=False)
model.fit(df_train[[column]], df_train[['y']])
y_predicted = model.predict(df_test[[column]])
predictions = y_predicted
score = r_score(df_test.y.values, y_predicted) * 100  
fig = plt.figure(1)
plt.plot(df_test.timestamp, predictions, color='blue')
plt.plot(df_test.timestamp, df_test.y, color='red')
plt.show()

fig = plt.figure(1)
plt.plot(df_test.timestamp, np.cumsum(predictions), color='blue')
plt.plot(df_test.timestamp, np.cumsum(df_test.y), color='red')
plt.show()
              
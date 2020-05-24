
import matplotlib.pyplot as plt
import pandas as pd

Linear_X_Test=pd.read_csv('Linear_X_Test.csv')
Linear_X_Train=pd.read_csv('Linear_X_Train.csv')
Linear_Y_Train=pd.read_csv('Linear_Y_Train.csv')

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(Linear_X_Train, Linear_Y_Train)

Y_pred = reg.predict(Linear_X_Test)

plt.scatter(Linear_X_Train,Linear_Y_Train,color='red')
plt.plot(Linear_X_Test, Y_pred, color = 'blue')
plt.title('The level students performace Vs devoted time')
plt.show()


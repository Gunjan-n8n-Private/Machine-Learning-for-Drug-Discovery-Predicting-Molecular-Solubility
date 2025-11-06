import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt #for the vizualization

###load the data ############################################
dataSet = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
#dataSet

###########################################

###data preparation ############################################

##data sepration as x-axis nd y-axis
y = dataSet['logS']
#y

x = dataSet.drop('logS',axis=1)
#print("x===>>",x)

##data splitting(80% for traning and 20% for testing)
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=100)

#print("x_train",x_train)
#print("x_test",x_test)

###########################################

###model building 

lr = LinearRegression()
lr.fit(x_train,y_train) # we train our model to the histroy traing data means we make a logic throught the existing history data

y_lr_train_pred = lr.predict(x_train) #we actually predict the value of y(train) with help of logic we already build in above last step means in above last step we make a logic and in this step we just give a train data to that logic to check models predicted valuse
y_lr_test_pred = lr.predict(x_test)


#y_lr_train_pred
#y_lr_test_pred

###########################################
###evaluate models performence

lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

lr_results = pd.DataFrame(['Linear regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
#lr_results

lr_results.columns = ['Method','Traning MSE','Training r2','Test MSE','Test r2'] #column name
lr_results 

###########################################
###data visualization of prediction model 

plt.figure(figsize=(10,6))

# Create scatter plots for actual vs predicted
plt.scatter(x=y_train, y=y_train, color='green', alpha=0.7, s=50, label='Actual Values')
plt.scatter(x=y_train, y=y_lr_train_pred, color='yellow', alpha=0.7, s=50, label='Predicted Values')

# Add reference line
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Values (Green)', color='green', fontsize=12)
plt.ylabel('Predicted Values (Yellow)', color='orange', fontsize=12)
plt.title('Actual vs Predicted Values\n(Green=Actual, Yellow=Predicted)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("Linear Regression Performance:")
print(f"Training R²: {lr_train_r2:.4f}")
print(f"Testing R²: {lr_test_r2:.4f}")
print(f"Training MSE: {lr_train_mse:.4f}")
print(f"Testing MSE: {lr_test_mse:.4f}")

###########################################

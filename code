#Import the libraries import streamlit as st
import datetime as dt import math import
pandas_datareader as web import numpy as np
import pandas as pd from
sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential from
keras.layers import Dense, LSTM import
matplotlib.pyplot as plt 
 
#To find closing price of particular day def
pred(symbol): 
 current_d=st.sidebar.date_input("Enter Date to view close price:") 
st.write('Close price of the entered date:') ds = symbol apple_quote2 = 
web.DataReader( ds , data_source='yahoo', start=current_d, end=c urrent_d) 
 
st.write(apple_quote2['Close']) 
#prediction Function def prediction(symbol, 
start, end): 
 ds = symbol start_d = start end_d = end df = 
web.DataReader(ds, data_source='yahoo', start=start_d, end = end_d) 
 #Show teh data 
tail_s = df.tail(10) 
st.table(tail_s) 
 #shape 
 
 #Visualize the closing price history 
 
 chart_data = pd.DataFrame( 
df, columns=['Close']) 
 
st.line_chart(chart_data) 
 
 
 
 
 
 #Create a new dataframe with only the 'Close column 
data = df.filter(['Close']) 
 #Convert the dataframe to a numpy array 
dataset = data.values 
 #Get the number of rows to train the model on 
training_data_len = math.ceil( len(dataset) * .8 ) 
 #Scale the data scaler = 
MinMaxScaler(feature_range=(0,1)) 
scaled_data = scaler.fit_transform(dataset) 
 
 #Create the training data set #Create the 
scaled training data set train_data = 
scaled_data[0:training_data_len , :] #Split the 
data into x_train and y_train data sets x_train = 
[] y_train = [] 
for i in range(60, 
len(train_data)): 
x_train.append(train_data[i-60:i, 0]) 
y_train.append(train_data[i,0]) 
 #Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train) 
 #Reshape the data x_train = np.reshape(x_train, 
(x_train.shape[0], x_train.shape[1], 1)) x_train.shape 
 
 #Build the LSTM model model = Sequential() model.add(LSTM(50, 
return_sequences=True, input_shape= (x_train.shape[1], 1))) 
model.add(LSTM(50, return_sequences= False)) model.add(Dense(25)) 
model.add(Dense(1)) 
 
 #Compile the model model.compile(optimizer='adam', 
loss='mean_squared_error') 
 #Train the model model.fit(x_train, y_train, 
batch_size=1, epochs=1) 
 #Create the testing dataset 
 #Create the new array containing scaled values from index 1802 to 2003 
test_data = scaled_data[training_data_len - 60: , :] 
 #Create the data sets x_test and 
y_test x_test = [] y_test = 
dataset[training_data_len:, :] for i
in range(60, len(test_data)): 
 x_test.append(test_data[i-60:i, 0]) 
 
 #Convert the data to a numpy array 
x_test = np.array(x_test) 
 
 #Reshape the data x_test = np.reshape(x_test, 
(x_test.shape[0], x_test.shape[1], 1 )) #Get 
the models prdicted price values predictions = 
model.predict(x_test) predictions = 
scaler.inverse_transform(predictions) 
 #Get the root mean squarred error (RMSE) rmse = 
np.sqrt( np.mean( predictions - y_test )**2 ) 
 #Plot the data train = 
data[:training_data_len] valid = 
data[training_data_len:] 
valid['Predictions'] = predictions 
 #Visualize the data plt.figure(figsize=(16,8)) 
plt.title('Model') 
 plt.xlabel('Date', fontsize=18) plt.ylabel('Close Price 
USD ($)', fontsize=18) plt.plot(train['Close']) 
plt.plot(valid[['Close', 'Predictions']]) 
plt.legend(['Train','Val','Predictions'], loc= 'lower right') 
plt.show() predict_chart_data = pd.DataFrame( valid, 
columns=['Close', 
'Predictions']) 
 
st.line_chart(predict_chart_data) 
 
 
 
 
 #Show the valid and predicted prics 
st.write('Close Price and Predicted Close ') 
st.write(valid) 
 
 #Get the quote to find the predicted CLOSE price p_end = 
st.sidebar.date_input("Enter Date to find the predicted Close:", dt.dat e(2021, 5, 
16)) apple_quote = web.DataReader( symbol , 
data_source='yahoo',start=start_d, end=p 
_end) 
 #Create a new dataframe new_df = apple_quote.filter(['Close']) #Get 
teh last 60 daysclosing price values and convert the dataframe to an array 
last_60_days = new_df[-60:].values 
 #Scale the data to be values between 0 and 1 last_60_days_scaled
= scaler.transform(last_60_days) 
 #Create an empty list 
 X_test = [] 
 #Append teh past 60 days 
 X_test.append(last_60_days_scaled) 
 #Convert the X_test data set to a numpy array 
 X_test = np.array(X_test) 
 #Reshape the data 
 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 
 #Get the predicted scaled price 
pred_price = model.predict(X_test) 
#undo the scaling pred_price = 
scaler.inverse_transform(pred_price) labels = 
['predicted price'] pred_price = 
pd.DataFrame(pred_price, columns=labels) 
 
 st.write("Predicted price for",p_end,":",pred_price) 
 
#working of streamlit code start_again = 0 end_again = 0 symbol = 
st.sidebar.text_input("Enter Stock Symbol:", value="AAPL") start = 
st.sidebar.date_input("Enter Start Date:", dt.date(2020, 3, 4)) end = 
st.sidebar.date_input("Enter End:", dt.date(2021, 5, 6)) 
 
#Main screen if symbol == '': st.write("Select Stock") else: 
st.write("Prediction for ", symbol) #Function for Prediction if start == 
start_again and end == end_again: print('repeated') else: 
 start_again = start end_again = end prediction(symbol, start, end) 
 
 # Function to get current value pred(symbol) 

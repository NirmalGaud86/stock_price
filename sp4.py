#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

# Load the dataset
df1 = pd.read_csv('jpmorgan_data.csv')
df1['Date'] = pd.to_datetime(df1['Date'], format="%d-%m-%Y")
df1.set_index('Date', inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df1['Close'].values.reshape(-1, 1))

# Get user input for date range
st.title('JP Morgan and Chase - Stock Price Prediction')
start_date = st.date_input('Enter the Start Date')
end_date = st.date_input('Enter the End Date')
predict_button = st.button('Predict the prices')

if predict_button:
    # Prepare the test data
    test_data = scaled_data[-60:, :]
    x_test = np.array([test_data])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Load the trained model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_test.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    model.load_weights('trained_model.h5')

    # Make predictions for user-defined date range
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions = []
    for _ in range(len(future_dates)):
        prediction = model.predict(x_test)
        predictions.append(scaler.inverse_transform(prediction)[0, 0])
        x_test = np.append(x_test[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    # Create a DataFrame of predictions
    predicted_prices = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
    predicted_prices.set_index('Date', inplace=True)

    # Display the predicted prices
    st.subheader('Predicted Stock Prices are:')
    st.dataframe(predicted_prices)

    # Plot the predicted prices
    st.subheader('Plot of Predicted Stock Prices')
    st.line_chart(predicted_prices['Predicted Price'])


# In[ ]:





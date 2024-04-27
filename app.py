import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(page_title="Stock  Prediction 📈")

# Add a header
st.header("StockAI 📈")

# Sidebar layout
st.sidebar.markdown("### StockAI")
st.sidebar.subheader("Enter Stock Ticker")
user_input = st.sidebar.text_input("", "")

# Fetching data and making predictions
if user_input:
    with st.spinner("Fetching data and making predictions..."):
        start = '2000-01-01'
        end = '2024-04-12'
        df = yf.download(user_input, start=start, end=end)

        # Splitting Data into Training and Testing
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        x_train = []
        y_train = []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i, 0])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Load the pre-trained model
        model = load_model('StockAI.h5')

        # Testing Part
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        X_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            X_test.append(input_data[i-100:i, 0])
            y_test.append(input_data[i, 0])

        X_test, y_test = np.array(X_test), np.array(y_test)
        y_predicted = model.predict(X_test)

        scaler = scaler.scale_
        scale_factor = 1/scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

    # Predicted Graph
    st.subheader('Predicted Price vs Original Price')
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Predicted Price vs Original Price', fontsize=16)
    plt.legend(fontsize=12)
    st.pyplot(plt)

    # Show Descriptive Statistics
    st.subheader('Data Description')
    st.write(df.describe())

    # Show Closing Price vs Time chart
    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Closing Price vs Time', fontsize=16)
    st.pyplot(fig)

    # Closing Price vs Time chart with 100MA
    st.markdown("### Closing Price vs Time Chart with 100MA")
    st.write("This chart shows the closing price of the stock over time, along with the 100-day moving average.")
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100MA')
    plt.plot(df['Close'], label='Closing Price')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Closing Price vs Time with 100MA', fontsize=16)
    plt.legend(fontsize=12)
    st.pyplot(fig)

    # Closing Price vs Time chart with 100MA & 200MA
    st.markdown("### Closing Price vs Time Chart with 100MA & 200MA")
    st.write("This chart shows the closing price of the stock over time, along with the 100-day and 200-day moving averages.")
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100MA')
    plt.plot(ma200, label='200MA')
    plt.plot(df['Close'], label='Closing Price')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('Closing Price vs Time with 100MA & 200MA', fontsize=16)
    plt.legend(fontsize=12)
    st.pyplot(fig)

else:
    st.warning("Please enter a stock ticker to see the prediction.")
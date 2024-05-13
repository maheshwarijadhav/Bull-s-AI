
# Bull's AI 📈

Bull's AI is a powerful stock prediction application built with Python and Streamlit. It utilizes advanced machine learning techniques to forecast future stock prices based on historical data. The application provides a user-friendly interface where users can input a stock ticker symbol and visualize the predicted stock prices alongside various other insightful charts and statistics.

## Features

- **Stock Price Prediction**: Enter a stock ticker symbol, and Bull's AI will fetch the historical data and make predictions for future stock prices using a pre-trained machine learning model.
- **Predicted Price vs Original Price Chart**: View a line chart comparing the predicted stock prices with the actual historical prices, allowing you to assess the model's accuracy.
- **Data Description**: Get a detailed overview of the stock's historical data, including descriptive statistics such as mean, standard deviation, and quartile values.
- **Closing Price vs Time Chart**: Visualize the closing prices of the stock over time, enabling you to identify trends and patterns.
- **Closing Price vs Time Chart with 100MA**: Explore the closing prices along with the 100-day moving average, a popular technical indicator used in stock analysis.
- **Closing Price vs Time Chart with 100MA & 200MA**: View the closing prices with both the 100-day and 200-day moving averages, providing additional insights for technical analysis.

## Installation

To run Bull's AI locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/your-username/bulls-ai.git
```

2. Navigate to the project directory:

```bash
cd bulls-ai
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. The application will open in your default web browser. If not, you can access it by visiting the URL provided in the terminal output.

## Usage

1. Enter a stock ticker symbol in the sidebar input field.
2. The application will fetch the historical data and make predictions for the entered stock.
3. Explore the various charts and statistics provided to gain insights into the stock's performance and the model's predictions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Streamlit](https://streamlit.io/) for providing an excellent framework for building data-centric applications.
- [yfinance](https://pypi.org/project/yfinance/) for convenient access to stock data.
- [Keras](https://keras.io/) for enabling the development of the machine learning model used in this application.
- [scikit-learn](https://scikit-learn.org/) for providing essential data preprocessing tools.

Stock Price Prediction Using Machine Learning
This project implements multiple machine learning models to predict stock prices based on historical data. The models used include:

LSTM (Long Short-Term Memory)

Random Forest

Support Vector Machine (SVM)

Linear Regression

The project demonstrates the process of data preprocessing, training models, evaluating performance, and visualizing the results. The code uses yfinance to fetch historical stock data and scikit-learn, Keras, and other libraries for model training and evaluation.

Requirements
Python 3.x

Libraries:

numpy

pandas

matplotlib

yfinance

sklearn

keras

tensorflow

To install the required libraries, run:

bash
Copy
Edit
pip install numpy pandas matplotlib yfinance scikit-learn keras tensorflow
Project Structure
Data Collection:

The stock data for the company is fetched using the yfinance library.

The stock symbol, date range, and historical data are used for predictions.

Data Preprocessing:

Data is preprocessed by normalizing the stock prices using the MinMaxScaler.

The dataset is split into training and testing sets (80% training and 20% testing).

Model Training:

Four different models are used for stock price prediction:

LSTM (Long Short-Term Memory): A deep learning model for time-series forecasting.

Random Forest Regressor: A machine learning model that uses multiple decision trees for regression tasks.

Support Vector Machine (SVM): A machine learning model using a radial basis function kernel for regression.

Linear Regression: A simple linear model for regression tasks.

Model Evaluation:

Each model’s performance is evaluated using Mean Squared Error (MSE).

Graphs are generated to visualize the predicted vs. actual stock prices for each model.

Bar charts and line plots help compare the performance of the models.

Model Saving:

The trained LSTM model is saved using model.save() for future use.

Usage
Download Stock Data:

The stock data for Google (GOOG) is fetched for the date range of January 1, 2012, to January 1, 2023. You can replace GOOG with any other stock symbol to fetch data for a different company.

Train Models:

The models are trained on historical stock data and predictions are made for the test data.

Evaluate Results:

The performance of each model is evaluated using Mean Squared Error (MSE) and visualized using line plots and bar charts.

Save Models:

The LSTM model is saved as Stock Predictions Model.keras for later use.

Example Outputs
LSTM Model:

Predicts future stock prices based on the past 100 days of stock data.

Results are plotted to compare the predicted vs. actual stock prices.

Random Forest:

Stock prices are predicted using features like Open, High, Low, and Volume.

The predicted stock prices are compared with the actual stock prices in both line and bar charts.

Support Vector Machine (SVM):

A support vector regression model is used to predict stock prices and compared with actual stock prices.

Linear Regression:

A linear regression model is used to predict the stock price, and the results are visualized.

Model Hyperparameters
LSTM:

units: Number of neurons in each LSTM layer.

Dropout: Used to prevent overfitting.

epochs: Number of training iterations.

Random Forest:

n_estimators: Number of trees in the forest.

Support Vector Machine (SVM):

C: Regularization parameter.

gamma: Kernel coefficient.

Linear Regression:

The model doesn't have hyperparameters, but it is trained using standard linear regression techniques.

Visualizations
Line Plot: Comparing predicted stock prices with actual stock prices.

Bar Chart: A visual comparison between predicted and actual stock prices for the first 20 data points.

Improved Plot: A more visually enhanced plot with markers and gridlines for clearer presentation.

Conclusion
This project demonstrates how various machine learning models can be applied to predict stock prices based on historical data. The comparison between LSTM, Random Forest, SVM, and Linear Regression allows users to explore different approaches to stock price forecasting.









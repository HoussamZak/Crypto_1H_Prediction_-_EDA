{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HNDtzvYZX7Pg"
      },
      "outputs": [],
      "source": [
        "#import pandas_datareader as webreader # Remote data access for pandas\n",
        "import math # Mathematical functions \n",
        "import numpy as np # Fundamental package for scientific computing with Python\n",
        "import pandas as pd # Additional functions for analysing and manipulating data\n",
        "from datetime import date, timedelta, datetime # Date Functions\n",
        "from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates\n",
        "import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data\n",
        "import matplotlib.dates as mdates # Formatting dates\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors\n",
        "from keras.models import Sequential # Deep learning library, used for neural networks\n",
        "from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers\n",
        "from keras.callbacks import EarlyStopping # EarlyStopping during model training\n",
        "from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data \n",
        "import seaborn as sns"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install python-binance"
      ],
      "metadata": {
        "id": "2upkEN1YYM-d"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v3mj3tYwX7QH"
      },
      "outputs": [],
      "source": [
        "from datetime import datetime\n",
        "from pandas_datareader.data import DataReader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lSs5MuQtX7QL"
      },
      "outputs": [],
      "source": [
        "# Data retrieval for cryptocurrencies BTCUSDT ETHUSDT XMRUSDT \n",
        "# Binance API credentials \n",
        "apikey = '2ZaEiNukiFtPHqnYdCGMANNimHdcCF0nvv4L9eYXMIdr4ovveQrca4NWKsZ7DxAy'\n",
        "secretkey = '9zjousoyG0t2wblcGkm0tVGLBfz6Woo3paakGh49Vhj0IxrGGXl332b1KQaxlJKY'\n",
        "\n",
        "from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager\n",
        "#API Initialization \n",
        "client = Client(apikey, secretkey)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "squpnHF3X7QN"
      },
      "outputs": [],
      "source": [
        "#Data collection: BTCUSDT \n",
        "#The idea is to focus on smaller time frames, specifically since the last crypto crash, mid-May 2021, to eliminate this skewing event\n",
        "\n",
        "klines = client.get_historical_klines(\"BTCUSDT\", Client.KLINE_INTERVAL_5MINUTE, \"1 Jun, 2021\", \"1 Dec, 2021\")\n",
        "\n",
        "# for candle in klines:\n",
        "#   print(candle)\n",
        "\n",
        "hist_df = pd.DataFrame(klines)\n",
        "hist_df.columns = (['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume','Nb of Trade', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume','Ignored'])\n",
        "# hist_df.drop(labels = ['TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignored', 'Quote Asset Volume'], inplace = True,axis = 1)\n",
        "# hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, unit='s')\n",
        "# hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')\n",
        "hist_df.drop(['Ignored', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Quote Asset Volume'], inplace= True, axis = 1)\n",
        "# hist_df['Open Time'] = hist_df.index\n",
        "\n",
        "hist_df = hist_df.reset_index().set_index('Open Time', drop=False)\n",
        "hist_df.index.name = None\n",
        "df = hist_df\n",
        "\n",
        "df.to_csv('BTCUSDT_5MIN_Jun2021_DEC2021')\n",
        "df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "At157IqIX7QT"
      },
      "outputs": [],
      "source": [
        "# Indexing Batches\n",
        "df_train = df.sort_values(by=['Open Time']).copy()\n",
        "\n",
        "# Saving a copy of the dates index, before we need to reset it to numbers\n",
        "date_index = df_train.index\n",
        "\n",
        "# We reset the index, so we can convert the date-index to a number-index\n",
        "df_train = df_train.reset_index(drop=True).copy()\n",
        "df_train.drop(columns=['index'], inplace=True, axis = 1)\n",
        "df_train.tail(5)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9lpTTQiRX7QX"
      },
      "outputs": [],
      "source": [
        "def prepare_data(df):\n",
        "\n",
        "    # List of considered Features\n",
        "    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']\n",
        "\n",
        "    print('FEATURE LIST')\n",
        "    print([f for f in FEATURES])\n",
        "\n",
        "    # Create the dataset with features and filter the data to the list of FEATURES\n",
        "    df_filter = df[FEATURES]\n",
        "    \n",
        "    # Convert the data to numpy values\n",
        "    np_filter_unscaled = np.array(df_filter)\n",
        "    #np_filter_unscaled = np.reshape(np_unscaled, (df_filter.shape[0], -1))\n",
        "    print(np_filter_unscaled.shape)\n",
        "\n",
        "    np_c_unscaled = np.array(df['Close']).reshape(-1, 1)\n",
        "    \n",
        "    return np_filter_unscaled, np_c_unscaled\n",
        "    \n",
        "np_filter_unscaled, np_c_unscaled = prepare_data(df_train)\n",
        "                                          \n",
        "# Creating a separate scaler that works on a single column for scaling predictions\n",
        "# Scale each feature to a range between 0 and 1\n",
        "scaler_train = MinMaxScaler()\n",
        "np_scaled = scaler_train.fit_transform(np_filter_unscaled)\n",
        "    \n",
        "# Create a separate scaler for a single column\n",
        "scaler_pred = MinMaxScaler()\n",
        "np_scaled_c = scaler_pred.fit_transform(np_c_unscaled)   "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "L8W_u897X7Qd"
      },
      "outputs": [],
      "source": [
        "# Set the input_sequence_length length - this is the timeframe used to make a single prediction\n",
        "input_sequence_length = 50\n",
        "# The output sequence length is the number of steps that the neural network predicts\n",
        "output_sequence_length = 10 #\n",
        "\n",
        "# Prediction Index\n",
        "index_Close = df_train.columns.get_loc(\"Close\")\n",
        "\n",
        "# Split the training data into train and train data sets\n",
        "# As a first step, we get the number of rows to train the model on 80% of the data \n",
        "train_data_length = math.ceil(np_scaled.shape[0] * 0.8)\n",
        "\n",
        "# Create the training and test data\n",
        "train_data = np_scaled[0:train_data_length, :]\n",
        "test_data = np_scaled[train_data_length - input_sequence_length:, :]\n",
        "\n",
        "# The RNN needs data with the format of [samples, time steps, features]\n",
        "# Here, we create N samples, input_sequence_length time steps per sample, and f features\n",
        "def partition_dataset(input_sequence_length, output_sequence_length, data):\n",
        "    x, y = [], []\n",
        "    data_len = data.shape[0]\n",
        "    for i in range(input_sequence_length, data_len - output_sequence_length):\n",
        "        x.append(data[i-input_sequence_length:i,:]) #contains input_sequence_length values 0-input_sequence_length * columns\n",
        "        y.append(data[i:i + output_sequence_length, index_Close]) #contains the prediction values for validation (3rd column = Close),  for single-step prediction\n",
        "    \n",
        "    # Convert the x and y to numpy arrays\n",
        "    x = np.array(x)\n",
        "    y = np.array(y)\n",
        "    return x, y\n",
        "\n",
        "# Generate training data and test data\n",
        "x_train, y_train = partition_dataset(input_sequence_length, output_sequence_length, train_data)\n",
        "x_test, y_test = partition_dataset(input_sequence_length, output_sequence_length, test_data)\n",
        "\n",
        "# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )\n",
        "print(x_train.shape, y_train.shape)\n",
        "print(x_test.shape, y_test.shape)\n",
        "\n",
        "# Validate that the prediction value and the input match up\n",
        "# The last close price of the second input sample should equal the first prediction value\n",
        "nrows = 3 # number of shifted plots\n",
        "fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 7))\n",
        "for i in range(nrows):\n",
        "    sns.lineplot(y = pd.DataFrame(x_train[i])[index_Close], x = range(input_sequence_length), ax = ax[i])\n",
        "    sns.lineplot(y = y_train[i], x = range(input_sequence_length -1, input_sequence_length + output_sequence_length - 1), ax = ax[i])\n",
        "plt.show"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x4bMq8PLX7Qi"
      },
      "source": [
        "Model Training "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oMIl6OdUX7Qp"
      },
      "outputs": [],
      "source": [
        "# Configure the neural network model\n",
        "model = Sequential()\n",
        "n_output_neurons = output_sequence_length\n",
        "\n",
        "# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables\n",
        "n_input_neurons = x_train.shape[1] * x_train.shape[2]\n",
        "print(n_input_neurons, x_train.shape[1], x_train.shape[2])\n",
        "model.add(LSTM(n_input_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) \n",
        "model.add(LSTM(n_input_neurons, return_sequences=False))\n",
        "model.add(Dense(5))\n",
        "model.add(Dense(n_output_neurons))\n",
        "\n",
        "# Compile the model\n",
        "model.compile(optimizer='adam', loss='mse')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Rr_JBEIoX7Qt",
        "outputId": "256dc0a2-3591-4808-ab6b-e29bd0adc0ed"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/100\n",
            "1314/1314 [==============================] - 21s 14ms/step - loss: 0.0011 - val_loss: 4.9977e-04\n",
            "Epoch 2/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 0.0010 - val_loss: 5.0721e-04\n",
            "Epoch 3/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.9738e-04 - val_loss: 4.8838e-04\n",
            "Epoch 4/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.9557e-04 - val_loss: 5.6211e-04\n",
            "Epoch 5/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.9502e-04 - val_loss: 4.8661e-04\n",
            "Epoch 6/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.9050e-04 - val_loss: 5.3197e-04\n",
            "Epoch 7/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8839e-04 - val_loss: 4.8952e-04\n",
            "Epoch 8/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8677e-04 - val_loss: 5.3136e-04\n",
            "Epoch 9/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8495e-04 - val_loss: 4.8728e-04\n",
            "Epoch 10/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8292e-04 - val_loss: 5.1610e-04\n",
            "Epoch 11/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8203e-04 - val_loss: 5.0595e-04\n",
            "Epoch 12/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8067e-04 - val_loss: 5.1588e-04\n",
            "Epoch 13/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.8148e-04 - val_loss: 4.8424e-04\n",
            "Epoch 14/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7910e-04 - val_loss: 4.8898e-04\n",
            "Epoch 15/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7967e-04 - val_loss: 4.8462e-04\n",
            "Epoch 16/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7804e-04 - val_loss: 5.0639e-04\n",
            "Epoch 17/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7731e-04 - val_loss: 5.2680e-04\n",
            "Epoch 18/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7698e-04 - val_loss: 4.8508e-04\n",
            "Epoch 19/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7545e-04 - val_loss: 4.8511e-04\n",
            "Epoch 20/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7573e-04 - val_loss: 5.0014e-04\n",
            "Epoch 21/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7526e-04 - val_loss: 4.8368e-04\n",
            "Epoch 22/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7441e-04 - val_loss: 4.8514e-04\n",
            "Epoch 23/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7338e-04 - val_loss: 4.8877e-04\n",
            "Epoch 24/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7379e-04 - val_loss: 4.8887e-04\n",
            "Epoch 25/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7377e-04 - val_loss: 4.9857e-04\n",
            "Epoch 26/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7037e-04 - val_loss: 4.9919e-04\n",
            "Epoch 27/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7132e-04 - val_loss: 4.8470e-04\n",
            "Epoch 28/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.7023e-04 - val_loss: 4.9291e-04\n",
            "Epoch 29/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6829e-04 - val_loss: 5.5187e-04\n",
            "Epoch 30/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6743e-04 - val_loss: 5.1965e-04\n",
            "Epoch 31/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6666e-04 - val_loss: 4.8687e-04\n",
            "Epoch 32/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6355e-04 - val_loss: 4.9677e-04\n",
            "Epoch 33/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6333e-04 - val_loss: 4.9408e-04\n",
            "Epoch 34/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6238e-04 - val_loss: 4.9127e-04\n",
            "Epoch 35/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.6280e-04 - val_loss: 4.9047e-04\n",
            "Epoch 36/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.5991e-04 - val_loss: 4.9992e-04\n",
            "Epoch 37/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.5571e-04 - val_loss: 4.9343e-04\n",
            "Epoch 38/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.5349e-04 - val_loss: 4.9730e-04\n",
            "Epoch 39/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.5137e-04 - val_loss: 5.0102e-04\n",
            "Epoch 40/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.5042e-04 - val_loss: 4.9661e-04\n",
            "Epoch 41/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.4488e-04 - val_loss: 5.0275e-04\n",
            "Epoch 42/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.4057e-04 - val_loss: 5.3510e-04\n",
            "Epoch 43/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.3726e-04 - val_loss: 5.0500e-04\n",
            "Epoch 44/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.2920e-04 - val_loss: 5.0187e-04\n",
            "Epoch 45/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.3518e-04 - val_loss: 5.0538e-04\n",
            "Epoch 46/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.2338e-04 - val_loss: 4.9595e-04\n",
            "Epoch 47/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.2068e-04 - val_loss: 4.9867e-04\n",
            "Epoch 48/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1653e-04 - val_loss: 5.2283e-04\n",
            "Epoch 49/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1146e-04 - val_loss: 5.4591e-04\n",
            "Epoch 50/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1504e-04 - val_loss: 5.4721e-04\n",
            "Epoch 51/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1254e-04 - val_loss: 5.2203e-04\n",
            "Epoch 52/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1246e-04 - val_loss: 5.0555e-04\n",
            "Epoch 53/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1339e-04 - val_loss: 5.1807e-04\n",
            "Epoch 54/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.0372e-04 - val_loss: 5.2066e-04\n",
            "Epoch 55/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.3094e-04 - val_loss: 4.9958e-04\n",
            "Epoch 56/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.1852e-04 - val_loss: 5.1530e-04\n",
            "Epoch 57/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 9.0315e-04 - val_loss: 5.1240e-04\n",
            "Epoch 58/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.9346e-04 - val_loss: 5.3507e-04\n",
            "Epoch 59/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.9492e-04 - val_loss: 5.7197e-04\n",
            "Epoch 60/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.9006e-04 - val_loss: 5.0015e-04\n",
            "Epoch 61/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.8804e-04 - val_loss: 4.9200e-04\n",
            "Epoch 62/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.8746e-04 - val_loss: 5.2708e-04\n",
            "Epoch 63/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.8045e-04 - val_loss: 5.8349e-04\n",
            "Epoch 64/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.8604e-04 - val_loss: 5.5818e-04\n",
            "Epoch 65/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.9042e-04 - val_loss: 5.2865e-04\n",
            "Epoch 66/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.7612e-04 - val_loss: 5.0625e-04\n",
            "Epoch 67/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.6799e-04 - val_loss: 6.0281e-04\n",
            "Epoch 68/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.7035e-04 - val_loss: 5.3350e-04\n",
            "Epoch 69/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.7229e-04 - val_loss: 5.1871e-04\n",
            "Epoch 70/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.6648e-04 - val_loss: 5.1861e-04\n",
            "Epoch 71/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.5203e-04 - val_loss: 5.3160e-04\n",
            "Epoch 72/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.5659e-04 - val_loss: 5.3064e-04\n",
            "Epoch 73/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.7580e-04 - val_loss: 5.6708e-04\n",
            "Epoch 74/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.4195e-04 - val_loss: 5.1544e-04\n",
            "Epoch 75/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.3308e-04 - val_loss: 5.1434e-04\n",
            "Epoch 76/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.2639e-04 - val_loss: 5.1896e-04\n",
            "Epoch 77/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.0661e-04 - val_loss: 5.0198e-04\n",
            "Epoch 78/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.0080e-04 - val_loss: 4.9982e-04\n",
            "Epoch 79/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 8.1086e-04 - val_loss: 5.1873e-04\n",
            "Epoch 80/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.8222e-04 - val_loss: 5.1187e-04\n",
            "Epoch 81/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.7701e-04 - val_loss: 5.0055e-04\n",
            "Epoch 82/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.6457e-04 - val_loss: 5.0453e-04\n",
            "Epoch 83/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.6575e-04 - val_loss: 5.0139e-04\n",
            "Epoch 84/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.5374e-04 - val_loss: 5.2020e-04\n",
            "Epoch 85/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.6814e-04 - val_loss: 5.0517e-04\n",
            "Epoch 86/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.3753e-04 - val_loss: 5.0022e-04\n",
            "Epoch 87/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.3240e-04 - val_loss: 5.1509e-04\n",
            "Epoch 88/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.4005e-04 - val_loss: 5.0559e-04\n",
            "Epoch 89/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.1995e-04 - val_loss: 5.0409e-04\n",
            "Epoch 90/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.1130e-04 - val_loss: 5.0289e-04\n",
            "Epoch 91/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 7.0419e-04 - val_loss: 5.0709e-04\n",
            "Epoch 92/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.9917e-04 - val_loss: 5.0466e-04\n",
            "Epoch 93/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.8678e-04 - val_loss: 5.0521e-04\n",
            "Epoch 94/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.7962e-04 - val_loss: 5.1947e-04\n",
            "Epoch 95/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.8308e-04 - val_loss: 4.9921e-04\n",
            "Epoch 96/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.6543e-04 - val_loss: 5.3033e-04\n",
            "Epoch 97/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.6444e-04 - val_loss: 5.1129e-04\n",
            "Epoch 98/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.4868e-04 - val_loss: 6.1301e-04\n",
            "Epoch 99/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.4358e-04 - val_loss: 5.0562e-04\n",
            "Epoch 100/100\n",
            "1314/1314 [==============================] - 17s 13ms/step - loss: 6.3515e-04 - val_loss: 5.0690e-04\n"
          ]
        }
      ],
      "source": [
        "# Training the model\n",
        "epochs = 100\n",
        "batch_size = 32\n",
        "early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)\n",
        "history = model.fit(x_train, y_train, \n",
        "                    batch_size=batch_size, \n",
        "                    epochs=epochs,\n",
        "                    validation_data=(x_test, y_test)\n",
        "                   )\n",
        "                    \n",
        "                    #callbacks=[early_stop])"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plot training & validation loss values\n",
        "fig, ax = plt.subplots(figsize=(10, 5), sharex=True)\n",
        "plt.plot(history.history[\"loss\"])\n",
        "plt.title(\"Model loss\")\n",
        "plt.ylabel(\"Loss\")\n",
        "plt.xlabel(\"Epoch\")\n",
        "ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))\n",
        "plt.legend([\"Train\", \"Test\"], loc=\"upper left\")\n",
        "plt.grid()\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "KPhy7O8aiuT0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 350
        },
        "outputId": "617be61e-57e9-45e8-b431-7b44cc41028b"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "K9T5iinHX7Qw"
      },
      "source": [
        "Model Performance Evaluation    "
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('Model_5MIN_JUN2021_DEC2021.h5')"
      ],
      "metadata": {
        "id": "1YUw4M6yg3RG"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "te7dOYe3X7Qz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6622485f-38fa-41ac-ca05-6a9a2b89703e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Median Absolute Error (MAE): 454.71\n",
            "Mean Absolute Percentage Error (MAPE): 1.47 %\n",
            "Median Absolute Percentage Error (MDAPE): 1.01 %\n"
          ]
        }
      ],
      "source": [
        "# Get the predicted values\n",
        "y_pred_scaled = model.predict(x_test)\n",
        "\n",
        "# Unscale the predicted values\n",
        "y_pred = scaler_pred.inverse_transform(y_pred_scaled)\n",
        "y_test_unscaled = scaler_pred.inverse_transform(y_test).reshape(-1, output_sequence_length)\n",
        "y_test_unscaled.shape\n",
        "\n",
        "# Mean Absolute Error (MAE)\n",
        "MAE = mean_absolute_error(y_test_unscaled, y_pred)\n",
        "print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')\n",
        "\n",
        "# Mean Absolute Percentage Error (MAPE)\n",
        "MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100\n",
        "print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')\n",
        "\n",
        "# Median Absolute Percentage Error (MDAPE)\n",
        "MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100\n",
        "print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wsJ349YSX7Q1"
      },
      "source": [
        "Multi Test Forecast"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "9ebER3a8X7Q2",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "outputId": "b55fbac7-3778-4758-a800-ccfb37a7cf6d"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 576x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 576x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "def plot_multi_test_forecast(i, s, x_test, y_test_unscaled, y_pred_unscaled): \n",
        "    \n",
        "    # reshape the testset into a one-dimensional array, so that it fits to the pred scaler\n",
        "    x_test_scaled_reshaped = np.array(pd.DataFrame(x_test[i])[index_Close]).reshape(-1, 1)\n",
        "    \n",
        "    # undo the scaling on the testset\n",
        "    df_test = pd.DataFrame(scaler_pred.inverse_transform(x_test_scaled_reshaped) )\n",
        "\n",
        "    # set the max index \n",
        "    test_max_index = df_test.shape[0]\n",
        "    pred_max_index = y_pred_unscaled[0].shape[0]\n",
        "    test_index = range(i, i + test_max_index)\n",
        "    pred_index = range(i + test_max_index, i + test_max_index + pred_max_index)\n",
        "    \n",
        "    # package y_pred_unscaled and y_test_unscaled into a dataframe with columns pred and true\n",
        "    data = pd.DataFrame(list(zip(y_pred_unscaled[s], y_test_unscaled[i])), columns=['pred', 'true']) #\n",
        "    \n",
        "    fig, ax = plt.subplots(figsize=(8, 4))\n",
        "    plt.title(f\"Predictions vs Ground Truth {pred_index}\", fontsize=12)\n",
        "    ax.set(ylabel =  \"BTC Price USDT\")\n",
        "    \n",
        "    sns.lineplot(data = df_test,  y = df_test[0], x=test_index, color=\"#039dfc\", linewidth=1.0, label='test')\n",
        "    sns.lineplot(data = data,  y='true', x=pred_index, color=\"g\", linewidth=1.0, label='true')\n",
        "    sns.lineplot(data = data,  y='pred', x=pred_index, color=\"r\", linewidth=1.0, label='pred')\n",
        "\n",
        "x_test_unscaled = scaler_pred.inverse_transform(np.array(pd.DataFrame(x_test[0])[index_Close]).reshape(-1, 1)) \n",
        "df_test = pd.DataFrame(x_test_unscaled)\n",
        "    \n",
        "for i in range(5, 7): #i is the starting point for the batch in the time-series\n",
        "    #data = pd.DataFrame(list(zip(y_pred[i], y_test_unscaled[i])), columns=['pred', 'true'], index=range(55,65))\n",
        "    plot_multi_test_forecast(i, i, x_test, y_test_unscaled, y_pred)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "46lUFqAOX7Q5"
      },
      "source": [
        "New Forecast "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "1xTnlSCuX7Q6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 281
        },
        "outputId": "89ed8ddd-328f-4e0c-b8a0-42b362bef5ce"
      },
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 576x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ],
      "source": [
        "def plot_new_multi_forecast(i_test, i_pred, x_test, y_pred_unscaled): \n",
        "    \n",
        "    # reshape the testset into a one-dimensional array, so that it fits to the pred scaler\n",
        "    x_test_scaled_reshaped = np.array(pd.DataFrame(x_test[i_test])[index_Close]).reshape(-1, 1)\n",
        "    \n",
        "    # undo the scaling on the testset\n",
        "    df_test = pd.DataFrame(scaler_pred.inverse_transform(x_test_scaled_reshaped) )\n",
        "\n",
        "    # set the max index \n",
        "    test_max_index = df_test.shape[0]\n",
        "    pred_max_index = y_pred_unscaled[0].shape[0]\n",
        "    test_index = range(i_test, i_test + test_max_index)\n",
        "    pred_index = range(i_test + test_max_index, i_test + test_max_index + pred_max_index)\n",
        "    \n",
        "    data = pd.DataFrame(list(zip(y_pred_unscaled[i_pred])), columns=['pred']) #\n",
        "    \n",
        "    fig, ax = plt.subplots(figsize=(8, 4))\n",
        "    plt.title(f\"Predictions vs Ground Truth {pred_index}\", fontsize=12)\n",
        "    ax.set(ylabel =  \"BTC Price USDT\")\n",
        "    \n",
        "    sns.lineplot(data = df_test,  y = df_test[0], x=test_index, color=\"#039dfc\", linewidth=1.0, label='test')\n",
        "    sns.lineplot(data = data,  y='pred', x=pred_index, color=\"r\", linewidth=1.0, label='pred')\n",
        "    \n",
        "# get the highest index from the x_test dataset\n",
        "index_max = x_test.shape[0]\n",
        "x_test_new = np_scaled[-51:-1,:].reshape(1,50,5)\n",
        "\n",
        "# undo the scaling of the predictions\n",
        "y_pred_scaled = model.predict(x_test_new)\n",
        "y_pred = scaler_pred.inverse_transform(y_pred_scaled)\n",
        "\n",
        "# plot the predictions\n",
        "plot_new_multi_forecast(0, 0, x_test_new, y_pred)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Actual Prediction\n",
        "\n",
        "def prediction(crypto):\n",
        "  #Choose between BTCUSDT ETHUSDT XMRUSDT\n",
        "\n",
        "  #Fetch Last Day 1HR Candles Data\n",
        "  klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_1HOUR, \"1 day ago UTC\")\n",
        "\n",
        "  pred_df = pd.DataFrame(klines)\n",
        "  pred_df.columns = (['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume','Nb of Trade', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume','Ignored'])\n",
        "  # hist_df.drop(labels = ['TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignored', 'Quote Asset Volume'], inplace = True,axis = 1)\n",
        "  pred_df['Close Time'] = pd.to_datetime(pred_df['Close Time']/1000, unit='s')\n",
        "  # hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, unit='s')\n",
        "  pred_df.drop(['Ignored', 'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Quote Asset Volume'], inplace= True, axis = 1)\n",
        "  # hist_df['Open Time'] = hist_df.index\n",
        "\n",
        "  # hist_df = hist_df.reset_index().set_index('Open Time', drop=False)\n",
        "  # hist_df.index.name = None\n",
        "\n",
        "  pred_df = pred_df.apply(pd.to_numeric)\n",
        "  pred = np_scaled[-51:-1,:].reshape(1,50,5)\n",
        "\n",
        "  # pred_df.to_csv('5H_avg_pred_df')\n",
        "  # pred_df\n",
        "\n",
        "  y_pred_scaled = model.predict(pred)\n",
        "  y_pred = scaler_pred.inverse_transform(y_pred_scaled)\n",
        "\n",
        "  print(y_pred)\n",
        "\n",
        "prediction(\"ETHUSDT\")\n"
      ],
      "metadata": {
        "id": "vvkib1Ndbzi7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ab5aa548-8aa4-47d9-a548-df56ba2336cf"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[29497.998 29476.223 29465.1   29465.508 29595.588 29664.814 29620.277\n",
            "  29647.643 29447.064 29479.328]]\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "interpreter": {
      "hash": "3bff544f8a8073c4167e9b2cc33d981654572eb557f5e5f174774942b3a02b19"
    },
    "kernelspec": {
      "display_name": "Python 3.8.5 64-bit",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    },
    "orig_nbformat": 4,
    "colab": {
      "name": "Model_5-15MIN_June2021_Dec2021_Taly.ipynb",
      "provenance": [],
      "machine_shape": "hm"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
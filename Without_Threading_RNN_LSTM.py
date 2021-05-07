import time
import signal
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import time
from tensorflow import keras
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model_type):
    print(f'Training a model: {model_type}')

    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ]

    model = Sequential()

    if model_type == 'rnn':
        model.add(SimpleRNN(units=1440, input_shape=(trainX.shape[1], trainX.shape[2])))
    elif model_type == 'lstm':
        model.add(LSTM(units=1440, input_shape=(trainX.shape[1], trainX.shape[2])))

    model.add(Dense(480))
    model.add(keras.layers.BatchNormalization())
    model.add(Activation('tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    """model.fit(
        trainX,
        trainY,
        epochs=50,
        batch_size=20,
        validation_data=(testX, testY),
        verbose=1,
        callbacks=callbacks,
    )

    # predict
    Y_Train_pred = model.predict(trainX)
    Y_Test_pred = model.predict(testX)

    train_MSE = mean_squared_error(trainY, Y_Train_pred)
    test_MSE = mean_squared_error(testY, Y_Test_pred)

    # you can also return values eg. the eval score
    return {'type': model_type, 'train_MSE': train_MSE, 'test_MSE': test_MSE}"""

start = time.time()
df = pd.read_csv("Train.csv", header=None)

index = [i for i in list(range(1440)) if i % 3 == 2]

Y_train = df[index]
df = df.values

# making history by using look-back to prediction next
def create_dataset(dataset, data_train, look_back=1):
    dataX, dataY = [], []
    print("Len:", len(dataset) - look_back - 1)
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), :]
        dataX.append(a)
        dataY.append(data_train[i + look_back, :])
    return np.array(dataX), np.array(dataY)


Y_train = np.array(Y_train)
df = np.array(df)

look_back = 10
trainX, trainY = create_dataset(df, Y_train, look_back=look_back)

# Split data into train & test
trainX, testX, trainY, testY = train_test_split(
    trainX, trainY, test_size=0.2, shuffle=False
)

model_types = ['rnn', 'lstm']
for model in model_types:
    print(train_model(model))
end = time.time()
print("time required: ", end - start)
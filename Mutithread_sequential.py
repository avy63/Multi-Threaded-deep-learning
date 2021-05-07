import tensorflow as tf
from tensorflow import keras
import multiprocessing
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import signal

# Fashion mnist data
fashion_mnist=keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test_class,y_test_class)=fashion_mnist.load_data()
X_valid_class,X_train_class=X_train_full[:5000]/255.0,X_train_full[5000:]/255.0
y_valid_class,y_train_class=y_train_full[:5000],y_train_full[5000:]
X_test_class=X_test_class/255.0
class_names=["T-Shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]
#Regression data:
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)
def train_model(model_type):
    if model_type=="classification":
        model=keras.models.Sequential()
        model.add(keras.layers.Flatten(input_shape=[28,28]))
        model.add(keras.layers.Dense(300,activation="relu"))
        model.add(keras.layers.Dense(100,activation="relu"))
        model.add(keras.layers.Dense(10,activation="softmax"))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        history=model.fit(X_train_class,y_train_class,epochs=30, validation_data=(X_valid_class,y_valid_class))
       
        mse_train=model.evaluate(X_train_class,y_train_class)
        mse_test=model.evaluate(X_test_class,y_test_class)
    elif model_type=="regression":
        
        model = keras.models.Sequential([
        keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
        keras.layers.Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer="sgd")
        history = model.fit(X_train_scaled, y_train, epochs=20,
        validation_data=(X_valid_scaled, y_valid))
        mse_train = model.evaluate(X_train_scaled, y_train)
        mse_test = model.evaluate(X_test_scaled, y_test)
        #X_new = X_test_scaled[:3] # pretend these are new instances
        #y_pred = model.predict(X_new)
        #X_new = X_test_scaled[:3] # pretend these are new instances
        #y_pred = model.predict(X_new)
    return {'type': model_type, 'train_MSE': mse_train, 'test_MSE': mse_test}


num_workers = 2
model_types = ['classification', 'regression']
start = time.time()
pool = multiprocessing.Pool(num_workers, init_worker)

scores = pool.map(train_model, model_types)
end = time.time()
print("time required: ", end - start)
print(scores)
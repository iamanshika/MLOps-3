from keras.models import Sequential
from keras.layers import Conv2D

def model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    return model
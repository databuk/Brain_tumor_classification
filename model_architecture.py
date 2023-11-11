from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(75, 75, 3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Dropout(0.1))


    model.add(Flatten())
    model.add(Dense(units= 32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
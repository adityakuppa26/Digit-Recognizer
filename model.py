import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

train_data = pd.read_csv('train.csv') 
test_data = pd.read_csv('test.csv')
train_data.head()

train_X = train_data.drop(columns = ['label'])
train_Y = train_data.label

# =============================================================================
# One-hot encode train_Y
# =============================================================================
train_Y = to_categorical(train_Y, num_classes=10)

# =============================================================================
# Normalization
# =============================================================================
train_X = train_X/255
test_data = test_data/255

# =============================================================================
# Convert to 3D image
# =============================================================================
train_X = train_X.values.reshape(-1, 28, 28, 1)
test_data = test_data.values.reshape(-1, 28, 28, 1)

# =============================================================================
# train-validation split
# =============================================================================
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2)

# =============================================================================
# define a model
# =============================================================================
model = Sequential()

model.add(Conv2D(kernel_size=(5,5), filters=32, strides=(1,1), padding='same', 
                 input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(kernel_size=(5,5), filters=64, strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(kernel_size=(3,3), filters=128, strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(kernel_size=(3,3), filters=256, strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

optimizer = Adam(lr=0.001)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

# =============================================================================
# Callbacks
# =============================================================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_lr=0.00001)

checkpoint = ModelCheckpoint(filepath='Model.weights.best.hdf5', save_best_only=True, verbose=1)


# =============================================================================
# Data Generator for augmentation
# =============================================================================
data_gen = ImageDataGenerator(rotation_range=20,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.15,
                              zoom_range=0.1,
                              vertical_flip=False,
                              horizontal_flip=False)

data_gen.fit(train_X)

# =============================================================================
# Training the model
# =============================================================================
history = model.fit_generator(data_gen.flow(train_X, y=train_Y, batch_size=64), 
                              epochs=20, steps_per_epoch=train_X.shape[0]//64, 
                              validation_data=(val_X, val_Y), callbacks=[reduce_lr, checkpoint])


# =============================================================================
# Predicting on test
# =============================================================================

model.load_weights('Model.weights.best.hdf5')

results=model.predict(test_file)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

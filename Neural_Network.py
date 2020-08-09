import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Creating a vector with lables data
imgs_chifre_Y = np.zeros(200)
imgs_L_Y = np.ones(200)
imgs_mao_aberta_Y = np.ones(200)*2
imgs_um_dedo_Y = np.ones(200)*3
imgs_V_Y = np.ones(200)*4

data_Y = np.concatenate((imgs_chifre_Y, imgs_L_Y, imgs_mao_aberta_Y, imgs_um_dedo_Y, imgs_V_Y), axis=0)

# Reading images from DataSet
imgs_chifre = []
imgs_L = []
imgs_mao_aberta = []
imgs_um_dedo = []
imgs_V = []

for i in range(1, 201):
    img = cv2.imread('Training/chifre/img'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
    imgs_chifre.append(img)
    img = cv2.imread('Training/L/img' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
    imgs_L.append(img)
    img = cv2.imread('Training/mao_aberta/img' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
    imgs_mao_aberta.append(img)
    img = cv2.imread('Training/um_dedo/img' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
    imgs_um_dedo.append(img)
    img = cv2.imread('Training/V/img' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
    imgs_V.append(img)

imgs_chifre = np.asarray(imgs_chifre)
imgs_L = np.asarray(imgs_L)
imgs_mao_aberta = np.asarray(imgs_mao_aberta)
imgs_um_dedo = np.asarray(imgs_um_dedo)
imgs_V = np.asarray(imgs_V)

data_X = np.concatenate((imgs_chifre, imgs_L, imgs_mao_aberta, imgs_um_dedo, imgs_V), axis=0)

# Splitting training set and testing set
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=1)

train_X = train_X.reshape(800, 256, 256, 1)
test_X = test_X.reshape(200, 256, 256, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# From training set splitting validation set
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=1)

# Settings
batch_size = 32
epochs = 2
num_classes = 5


# Model settings
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(256,256,1)))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])

model.summary()

# Fitting the model
model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_Y))

# Saving the model
model.save("Model.h5py")

# Evaluating the model
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


predicted_classes = model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
correct = np.where(predicted_classes==test_Y)[0]
print("Found "+str(len(correct))+" correct labels")
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(256,256), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found "+str(len(incorrect))+" incorrect labels")
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))

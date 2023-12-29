import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import warnings
import joblib
warnings.filterwarnings("ignore")
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.applications.xception import Xception
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
data = []
labels= []
data_1=os.listdir("../cell_images/Parasitized/")
for i in data_1:
    try:
        image = cv2.imread("../cell_images/Parasitized/"+i)
        image_from_array= Image.fromarray(image , "RGB")
        size_image =image_from_array.resize((50,50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")
Uninfected = os.listdir("../cell_images/Uninfected/")
for b in Uninfected:
    try :
        image = cv2.imread("../cell_images/Uninfected/"+b)
        array_image=Image.fromarray(image,"RGB")
        size_image=array_image.resize((50,50))
        resize45= size_image.rotate(15)
        resize75 = size_image.rotate(25)
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")
Cells =np.array(data)
labels =np.array(labels)
print(labels.shape)
print(Cells.shape)
s=np.arange(Cells.shape[0])
np.random.shuffle(s)
len_data = len(Cells)
Cells=Cells[s]
labels =labels[s]
labels =keras.utils.to_categorical(labels)
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
 
Cells=Cells/255
model.fit(Cells,labels,batch_size=50,epochs=10,verbose=1)
model.save("my_model.h5")
 
joblib.load("model")
joblib.dump(model,"model")
model.save("model111.h5")
model11=load_model("model111.h5")
model11.predict(Cells[73].reshape(1,50,50,3))
blur=cv2.blur(Cells[1000].rotate(45),(5,5))
joblib.dump(model,"Malaria Cell model")
model1=Xception()
modl= keras.applications.vgg16.VGG16()
modl.summary()
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
train_img=ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.2,horizontal_flip=True)
train_images=train_img.flow_from_directory("../cell_images/Parasitized/",target_size=(64,64,3),batch_size=32)
 
7) pneuomonia.py
The model type that we use is Sequential. The activation function we use is the ReLU. Flatten serves as a connection between the convolution and dense layers. ‘Dense’ is the layer type we will use in our output layer. 
The activation is ‘softmax’. The model will then make its prediction based on which option has the highest probability. The trained model is stored in my_model.h5
 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(os.listdir("../input/chest_xray/chest_xray"))
print(os.listdir("../input/chest_xray/chest_xray/train"))
print(os.listdir("../input/chest_xray/chest_xray/train/"))
img_name = 'NORMAL2-IM-0588-0001.jpeg'
img_normal = load_img('../input/chest_xray/chest_xray/train/NORMAL/' + img_name)
print('NORMAL')
plt.imshow(img_normal)
plt.show()
 
img_name = 'person63_bacteria_306.jpeg'
img_pneumonia = load_img('../input/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)
print('PNEUMONIA')
plt.imshow(img_pneumonia)
plt.show()
 
img_width, img_height = 150, 150
train_data_dir = '../input/chest_xray/chest_xray/train'
validation_data_dir = '../input/chest_xray/chest_xray/val'
test_data_dir = '../input/chest_xray/chest_xray/test'
nb_train_samples = 5217
nb_validation_samples = 17
epochs = 20
batch_size = 16
 
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.layers
 
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
model.save('my_model.h5')
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

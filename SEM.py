# Importing related libraries
import numpy as np
import pandas as pd
import keras
import cv2
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.applications import VGG19
# Adjusting Import Directory
cwd = os.getcwd()

## Predefined functions
 
# Sharpfactor is an int betweeen 0 and 1 determines how much high the result is to be sharpened.
def sharpen(img,sharpfactor):
    threshold = 0.5
    blurred = cv2.GaussianBlur(img,ksize = (3,3),sigmaX=0)
    masked = np.abs(img - blurred) < threshold
    image = cv2.addWeighted(blurred, -sharpfactor, img , (1+sharpfactor), gamma = 0)
    binmasked = masked.astype(bool)
    image = image * (binmasked.astype(image.dtype))
    return image
                          
                          
# Loading train images
image_path = cwd + "SEM/"          # path of train images
images = glob.glob(image_path +"*.png")                        # selecting images
images.sort()                                                  # sorting images
final_images=[]
for img in images:    
    image = cv2.imread(img,0)                          # loading images
    image = cv2.resize(image,(500,300))                # resizing images
    image = image / np.max(image)                      # Normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    image = sharpen(image,0.8)
    image = image.tolist()
    final_images.append(image)
    
# loading train labels
labels = pd.read_excel("SEM Labels.xlsx")
final_labels = labels.values.tolist()
# split data
img_train, img_test, lbl_train, lbl_test = train_test_split(final_images, final_labels,
                                                            test_size=0.2, random_state=42, stratify=final_labels)
                                     
                                                            
# Creating Convolutional Model
def BuildModel():
    CNT_Model = Sequential()
    
    reg = regularizers.l1_l2(l1 = 0.001, l2 = 0.002)
    
    CNT_Model.add(Conv2D(20,5, activation = 'relu',strides=3, padding = 'same',kernel_regularizer = None, input_shape =(300,500,1)))
    CNT_Model.add(Conv2D(20,5, activation = 'relu',strides=2, padding = 'same', kernel_regularizer = None))
    CNT_Model.add(MaxPooling2D(pool_size = (2,2)))
    #CNT_Model.add(Dropout(0.8))
    CNT_Model.add(Conv2D(40,5, activation = 'relu', padding = 'same', kernel_regularizer = None))
    CNT_Model.add(MaxPooling2D(pool_size = (2,2)))
    CNT_Model.add(Conv2D(40,5, activation = 'relu', padding = 'same', kernel_regularizer = None))
    #CNT_Model.add(Dropout(0.5))
    CNT_Model.add(Flatten())
    CNT_Model.add(Dense(15,activation = 'relu', kernel_regularizer =None))       
    #CNT_Model.add(Dropout(0.5))
    #CNT_Model.add(BatchNormalization())
    CNT_Model.add(Dense(8,activation = 'relu', kernel_regularizer = None))
    #CNT_Model.add(BatchNormalization())
    CNT_Model.add(Dense(2,activation='sigmoid'))

    Optimizer = keras.optimizers.SGD(lr=0.01, momentum = 0.5, decay = 1e-5, nesterov = True)
   
    CNT_Model.compile(optimizer=Optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return CNT_Model
CNT_Model = BuildModel()
CNT_Model.summary()

# Training model
img_train = np.reshape(img_train,(len(img_train),300,500,1))

ann_hist = CNT_Model.fit(np.array(img_train), np.array(lbl_train), validation_split = 0.1, batch_size = 3, epochs = 100)


# Model Evaluation
img_test = np.reshape(img_test,(len(img_test),300,500,1))
test_loss, test_acc = CNT_Model.evaluate(np.array(img_test), np.array(lbl_test), batch_size = 3)
labels_predicted = CNT_Model.predict(np.array(img_test))
lbl= np.argmax(labels_predicted, axis=1)
print('Test loss is',test_loss)
print('Test accuracy is',test_acc)

# Plotting results
def ploth(ann_hist):
    plt.figure()
    hist = ann_hist.history
    loss = hist['loss']
    acc = hist['acc']
    plt.plot(acc)
    plt.plot(loss)
    plt.figure()
    val_loss = hist['val_loss']
    val_acc = hist['val_acc']
    plt.plot(val_loss)
    plt.plot(val_acc)
    plt.xlabel=('Epoch')
    plt.ylabel=('Loss')
ploth(ann_hist)


# K-fold cross validation
k = 4
num_val_samples = len(img_train) // k
epochs = 100
all_scores = []

for i in range(k):
    print('Training process of fold #{0} is going on.'.format(i))
    val_data = img_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = lbl_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = img_train[:num_val_samples * k] + img_train[num_val_samples * (k + 1):]
    
    partial_train_targets = lbl_train[:num_val_samples * k] + lbl_train[num_val_samples * (k + 1):]
    
    model = BuildModel()
    
    partial_train_data = np.reshape(partial_train_data,(len(partial_train_data),300,500,1))
    
    model.fit(np.array(partial_train_data), np.array(partial_train_targets),
    epochs=epochs, batch_size=3, verbose=0)
    
    val_data = np.reshape(val_data,(len(val_data),300,500,1))

    val_mse, val_mae = model.evaluate(np.array(val_data), np.array(val_targets))
    all_scores.append(val_mse)
total_error = np.average(all_scores)
print('The validation mean error after k-fold CV is',total_error)


# Transfer Learning

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(300,500,3))
conv_base.summary()


# Creating Pretrained Network
TL_model = Sequential()
TL_model.add(conv_base)
TL_model.add(Flatten())
TL_model.add(Dense(10, activation='relu'))
TL_model.add(Dense(2, activation='sigmoid'))

conv_base.trainable = True

# Fine Tuning
#for layer in conv_base.layers:
#    if layer.name == 'block5_conv1':
#        layer.trainable = True
#    else:
#        layer.trainable = False

Optimizer = keras.optimizers.SGD(lr=0.01, momentum = 0.5, decay = 1e-5, nesterov = True)
   
TL_model.compile(optimizer=Optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
TL_model.summary()

# Training with Data Augmentation
base_dir = cwd + "SEM/"

train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='wrap')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
base_dir,
batch_size=20,
class_mode='binary')

history = TL_model.fit_generator(
train_generator,
steps_per_epoch=100,
epochs=30)

ploth(history)



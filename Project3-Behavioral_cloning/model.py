import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D,MaxPooling2D,Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

from keras.models import model_from_json


lines= []
def read_lines(driving_log_path, extract_lines):
    with open(driving_log_path) as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            extract_lines.append(line)
    return extract_lines
lines =read_lines("./data/data/driving_log.csv",lines)
#print("lines",len(lines))
lines=lines[6300:]
print(lines)

train_samples,validation_samples =train_test_split(lines,test_size=0.2)
###Preprocessing data...
###original input shape is (160,320,3)
###1.resize.. width=200
###2.crop..   heigth=66
###3.Normalized

def generator(lines,batch_size=32):
    num_lines=len(lines)
    while 1: ## Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0,num_lines,batch_size):
            batch_lines= lines[offset:offset+batch_size]
            images=[]
            angles=[]
            for batch_line in batch_lines:
                name="./data/data/IMG/"+batch_line[0].split("/")[-1]
                #name="/Users/likangning/Desktop/raw_data/"+batch_line[0]
                #name=batch_line[0]
                center_image =cv2.imread(name)
                center_angle= float(batch_line[3])
                images.append(center_image)
                angles.append(center_angle)
            ###trim image to only see section with road
            ###convert the image to np.array
            x_train= np.array(images)
            y_train= np.array(angles)
            ###yield in generator means return in normal function
        yield shuffle(x_train,y_train)


### Compile and train the model using the generator function

train_generator= generator(train_samples,batch_size=32)
validation_generator= generator(validation_samples,batch_size=32)

def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image,(66,200))
def normalize(image):
    ####Normalize the input between [-0.5,0.5]
    return image /255. -0.5

model1= Sequential()
model1.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
model1.add(Lambda(resize))
model1.add(Lambda(normalize))
model1.add(Convolution2D(24,5,5,border_mode="valid",subsample=(2,2)))
    #model1.add(MaxPooling2D(2,2))
model1.add(Activation("relu"))
model1.add(Convolution2D(36,5,5,border_mode="valid",subsample=(2,2)))
    #model1.add(MaxPooling2D(2,2))
model1.add(Activation("relu"))
model1.add(Convolution2D(48,5,5,border_mode="valid",subsample=(2,2)))
    #model1.add(MaxPooling2D(2,2))
model1.add(Activation("relu"))
model1.add(Convolution2D(64,3,3,border_mode="valid",subsample=(1,1)))
    #model1.add(MaxPooling2D(2,2))
model1.add(Activation("relu"))
model1.add(Flatten())
model1.add(Dropout(0.5))
model1.add(Activation("relu"))
    
model1.add(Dense(200))
model1.add(Dropout(0.2))
model1.add(Activation("relu"))
    
model1.add(Dense(50))
model1.add(Dropout(0.2))
model1.add(Activation("relu"))

model1.add(Dense(10))
model1.add(Dropout(0.2))
model1.add(Activation("relu"))
    
model1.add(Dense(1))
model1.summary()
model1.compile(loss="mse",optimizer=Adam(lr= 0.001))


batch_size=32
nb_epoch=1

checkpointer= ModelCheckpoint(filepath="./Checkpointer/comma-4c.{epoch:02d}-{val_loss:.2f}.hdf5",verbose=1,save_best_only=False)

model1.fit_generator(train_generator,
                     samples_per_epoch=len(train_samples),
                     validation_data=validation_generator,
                     nb_val_samples=len(validation_samples),
                     nb_epoch=nb_epoch,
                     callbacks=[checkpointer],
                     verbose=1)




model_json=model1.to_json()
with open("model.json","w") as json_file:
     json_file.write(model_json)


model1.save("model.h5")
print("done")




























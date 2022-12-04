#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2


# In[10]:


labels = os.listdir("../DS_340/Driver-Drowsiness-Dataset/train")


# In[11]:


#file = '.DS_Store'
#location = "../DS_340/Driver-Drowsiness-Dataset/train"
#path = os.path.join(location, file)

#print(labels)


# In[12]:


#os.remove(path)


# In[13]:


labels


# In[14]:


#closed
import matplotlib.pyplot as plt
plt.imshow(plt.imread("../DS_340/Driver-Drowsiness-Dataset/train/Closed/_10.jpg"))


# In[15]:


#open
import matplotlib.pyplot as plt
plt.imshow(plt.imread("../DS_340/Driver-Drowsiness-Dataset/train/Open/_100.jpg"))


# In[16]:


#image array (yawn)
a = plt.imread("../DS_340/Driver-Drowsiness-Dataset/train/yawn/10.jpg")
#image shape
a.shape


# In[17]:


#no yawn
plt.imshow(plt.imread("../DS_340/Driver-Drowsiness-Dataset/train/no_yawn/1028.jpg"))


# In[18]:


#yawn
plt.imshow(plt.imread("../DS_340/Driver-Drowsiness-dataset/train/yawn/104.jpg"))


# In[22]:


#get rid of background in yawns eliminate noise
def face_for_yawn(direc="../DS_340/Driver-Drowsiness-Dataset/train", face_cas_path="../DS_340/prediction-images/haarcascade_frontalface_default.xml"):
    yaw_no = []
    #yawn-0
    #no_yawn-1
    IMG_SIZE = 32
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no


yawn_no_yawn = face_for_yawn()


# In[23]:


#get rid of backgrounds in open/closed
def get_data(dir_path="../DS_340/Driver-Drowsiness-Dataset/train/", face_cas="../DS_340/prediction-images/haarcascade_frontalface_default.xml", eye_cas="../DS_340/prediction-images/haarcascade.xml"):
    labels = ['Closed', 'Open']
    #close-2
    #open-3
    IMG_SIZE = 32
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num +=2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

data_train= get_data()


# In[24]:


#extend data and convert array
def append_data():
#     total_data = []
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)

#new variable to store 
new_data= append_data()


# In[25]:


#separate label and features
X = []
Y = []
for feature, label in new_data:
    X.append(feature)
    Y.append(label)


# In[26]:


#need to reshape the data

X = np.array(X)
X = X.reshape(-1, 32, 32, 3)
X.shape


# In[28]:


from sklearn.preprocessing import LabelBinarizer
#turns list into a matrix, where the number of columns in 
#the target matrix is exactly as many as unique value in the input set
label_bin = LabelBinarizer()
Y = label_bin.fit_transform(Y)


# In[52]:


#label the array
Y = np.array(Y)
Y.shape


# In[29]:


#split the train and test data
from sklearn.model_selection import train_test_split
test_size = 0.30
#With random_state=42, we get the same train and test sets across different executions,
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42, test_size=test_size)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#len(X_test)
#Y_test


# In[40]:


#!pip install tensorflow==2.3.1
#!pip install keras==2.4.3


# In[30]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
#from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import keras


# In[65]:


#data augmentation
#Keras image data generator class is also used to carry out data augmentation where we 
#aim to gain the overall increment in the generalization of the model.

#Operations such as rotations, translations, shearin, scale changes, and horizontal flips are 
#carried out randomly in data augmentation using an image data generator.
#train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
#test_generator = ImageDataGenerator(rescale=1/255)

#train_generator = train_generator.flow(np.array(X_train), Y_train, shuffle=False)
#test_generator = test_generator.flow(np.array(X_test), Y_test, shuffle=False)


# In[35]:


#create model
def driver_drowsiness_detection_model(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), name='conv1', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=42)))
    #it is a process to make neural networks faster and more stable through adding extra layers in a deep neural network.
    model.add(BatchNormalization())
    #do for all layers
    
    model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), name='conv2', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    #helps prevent overfitting
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), name='conv3', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), name='conv4', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv5', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv6', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), name='conv7', activation='relu', 
                     kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc3'))
    
    optimizer = Adam(0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[36]:


model= driver_drowsiness_detection_model(input_shape=(32, 32, 3))
model.summary()


# In[37]:


#training the model
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.2, horizontal_flip=True)
history = model.fit(aug.flow(X_train, Y_train, batch_size=128), batch_size=128, epochs=200, validation_data=(X_test, Y_test))
#batch_size=128
#history = model.fit(train_generator, batch_size=128, epochs=200, validation_data=test_generator, shuffle=True, steps_per_epoch=int(round(len(train_generator)/batch_size)), validation_steps=len(test_generator)/batch_size)


# In[38]:


#plot history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
#graph 1
#plotting accuracy aganist num of epochs for train and test sets
plt.plot(epochs, accuracy, "b", label="Trainning Accuracy")
plt.plot(epochs, val_accuracy, "r", label="Testing Accuracy")
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#graph 2
#plotting loss values aganist num of epochs for train and test sets
plt.plot(epochs, loss, "b", label="Trainning Loss")
plt.plot(epochs, val_loss, "r", label="Testing Loss")
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[39]:


#model.save("drowiness_new6.h5")


# In[40]:


model.save("drowiness_graphs.model")


# In[41]:


#evaluating model on test set
pred = model.evaluate(X_test, Y_test)
print(f'Test Set Accuracy: {pred[1]}')
print(f'Test Set Loss: {pred[0]}')


# In[42]:


#prediction
predict_x= model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)


# In[43]:


prediction=classes_x.reshape(-1,1)


# In[44]:


prediction.shape


# In[45]:


type(prediction)


# In[46]:


#
import numpy as np
n_values = np.max(prediction) + 1
pred=np.eye(n_values)[prediction]
pred


# In[47]:


print(Y_test.shape)
print(pred.shape)
pred=np.reshape(pred, (578*1, 4))
pred


# In[48]:


from sklearn.metrics import accuracy_score
#Generate the confusion matrix
cf_matrix=confusion_matrix(Y_test.argmax(axis=1), pred.argmax(axis=1))
print(cf_matrix)
print(accuracy_score(Y_test.argmax(axis=1), pred.argmax(axis=1)))


# In[49]:


#add labels to graph w/ labels
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
ax.xaxis.set_ticklabels(['yawn','no_yawn','Closed','Open'])
ax.yaxis.set_ticklabels(['yawn','no_yawn','Closed','Open'])


# In[50]:


model.save("drowiness_matrix.h5")


# In[51]:


labels_new = ["yawn", "no_yawn", "Closed", "Open"]


# In[52]:


#create the classification report
#from sklearn.metrics import classification_report
print(classification_report(np.argmax(Y_test, axis=1), prediction, target_names=labels_new))


# In[54]:


#predicitng function
labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 32
def prepare(filepath, face_cas="../DS_340/prediction-images/haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("./drowiness_matrix.h5")


# In[55]:


#no_yawn -1 
prediction = model.predict([prepare("../DS_340/Driver-Drowsiness-Dataset/train/no_yawn/1067.jpg")])
np.argmax(prediction)


# In[56]:


#closed
prediction = model.predict([prepare("../DS_340/Driver-Drowsiness-Dataset/train/Closed/_101.jpg")])
np.argmax(prediction)


# In[58]:


prediction = model.predict([prepare("../DS_340/Driver-Drowsiness-Dataset/train/Open/_104.jpg")])
np.argmax(prediction)


# In[59]:


prediction = model.predict([prepare("../DS_340/Driver-Drowsiness-Dataset/train/yawn/113.jpg")])
np.argmax(prediction)


# In[ ]:





import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import random
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.metrics import Recall,Precision

save_path = r'D:\Models\Braintumor'
folder = r'D:\BrainTumorDetector\dataset'
test_fol = r'D:\BrainTumorDetector\Testing'
tf_callback = r'D:\Models\Braintumor\callbacks'

def get_data_from_directory(folder):
    data = image_dataset_from_directory(folder,label_mode='categorical')
    return data


def describe_data(data):
    data_iter = data.as_numpy_iterator()
    batch = data_iter.next()
    print('Visualising data:\n')
    fig,axs = plt.subplots(ncols=4, figsize=(20,20))
    for id,img in enumerate(batch[0][:4]):
        axs[id].imshow(img.astype(int))
        axs[id].title.set_text(batch[1][id])
    print('\nLength of data:',len(data))
    print('\nImage shape:',batch[0].shape)

    
def split_data(data):
    train_size = int(len(data)*0.75)
    val_size = int(len(data)*0.10)+2
    test_size = int(len(data)*0.15)
    print(train_size,':',val_size,':',test_size)
    train = data.take(train_size)
    validation = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    return train,validation,test


def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu',input_shape=(256,256,3)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(filters=128,kernel_size=(3,3),strides=1,activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=4,activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def model_train_evaluation(model,train,validation,test):
    callbacks = tf.keras.callbacks.TensorBoard(log_dir=tf_callback)
    results = model.fit(train, epochs=15, validation_data=validation, callbacks=[callbacks])
    precision = Precision()
    recall = Recall()
    
    for test_batch in test:
        X,y = test_batch
        pred = model.predict(X)
        precision.update_state(y,pred)
        recall.update_state(y,pred)
    
    f1_score = (2*precision.result()*recall.result())/(precision.result()+recall.result())
    print('precision:{}\nrecall:{}\nf1-score:{}'.format(precision.result(),recall.result(),f1_score))
    
    model.save(os.path.join(save_path,'ICmodel_1.h5'))
    return model,results

def visualize_results(results):
    fig = plt.figure()
    plt.plot(results.history['loss'],color='red',label='Loss')
    plt.plot(results.history['val_loss'],color='blue',label='Val_loss')
    fig.suptitle('Loss Graph',fontsize=20)
    plt.legend()
    plt.plot(results.history['accuracy'],color='red',label='Accuracy')
    plt.plot(results.history['val_accuracy'],color='blue',label='Val_accuracy')
    fig.suptitle('Accuracy Graph',fontsize=20)
    plt.legend()
    plt.show()

def test_model(model):
    test_dirs = ['glioma','meningioma','notumor','pituitary']
    directory = random.choice(test_dirs)
    allimgs = os.listdir(os.path.join(test_fol,directory))
    test_img = random.choice(allimgs)
    test_img = os.path.join(test_fol,directory,test_img)
    print(test_img)
    timg = cv2.imread(test_img)
    timg = tf.image.resize(timg,(256,256))
    print(timg.shape)
    
    predict = model.predict(np.expand_dims(timg,axis=0))
    print(np.argmax(predict))
    

data = get_data_from_directory(folder)
describe_data(data)
train,validation,test = split_data(data)
model = create_model()
model,results = model_train_evaluation(model,train,validation,test)
test_model(model)
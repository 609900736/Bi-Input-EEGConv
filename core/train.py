#coding:utf-8

import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint

import core.models as models
import core.utils as utils



def make_checkpoint(filepath):
    if not os.path.exists('./model'): #判断是否存在
        os.makedirs('./model') #不存在则创建
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                                   save_best_only=True)
    return checkpointer

def train(data, label, batch_size=1, model=None, optimizer='adam',
          loss='sparse_categorical_crossentropy', metrics=['accuracy'],
          epochs=300, validation_split=0.0, validation_data=None):
    if not (type(model) is tf.keras.Model):
        raise(ValueError('wrong type of model'))

    tm = time.localtime()
    filepath = os.path.join('./model',str(tm.tm_year)+'-'+str(tm.tm_mon)+
                            '-'+str(tm.tm_mday)+'-'+str(tm.tm_hour)+'-'+
                            str(tm.tm_min)+'-'+str(tm.tm_sec)+'-checkpoint.h5')
    checkpointer = make_checkpoint(filepath)

    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    history = model.fit(x=data,y=label,batch_size=batch_size,epochs=epochs,
              callbacks=[checkpointer],validation_split=validation_split,
              validation_data=validation_data)

    model.load_weights(filepath)
    model.evaluate(data,label)

    filepath = os.path.join('./model',str(tm.tm_year)+'-'+str(tm.tm_mon)+
                            '-'+str(tm.tm_mday)+'-'+str(tm.tm_hour)+'-'+
                            str(tm.tm_min)+'-'+str(tm.tm_sec)+'-model.h5')
    model.save(filepath)

    # Plot training & validation accuracy values
    plt.figure(0)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return model

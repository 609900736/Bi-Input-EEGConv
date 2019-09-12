#coding:utf-8
import core.model as model
import core.train as train
import core.utils as utils
import scipy.io as sio

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def multi_input_model():
    """构建多输入模型"""
    input1_ = tf.keras.Input(shape=(100, 1), name='input1')
    input2_ = tf.keras.Input(shape=(50, 1), name='input2')
 
    x1 = tf.keras.layers.Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input1_)
    x1 = tf.keras.layers.MaxPool1D(pool_size=10, strides=10)(x1)
 
    x2 = tf.keras.layers.Conv1D(16, kernel_size=3, strides=1, activation='relu', padding='same')(input2_)
    x2 = tf.keras.layers.MaxPool1D(pool_size=5, strides=5)(x2)
 
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Flatten()(x)
 
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output_ = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
 
    model = tf.keras.Model(inputs=[input1_, input2_], outputs=[output_])
    model.summary()
 
    return model
 
if __name__ == '__main__':
    # 产生训练数据
    x1 = np.random.rand(100, 100, 1)
    x2 = np.random.rand(100, 50, 1)
    # 产生标签
    y = np.random.randint(0, 2, (100,))
 
    model = multi_input_model()
    # 保存模型图
    tf.keras.utils.plot_model(model, 'Multi_input_model.png')
 
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    history = model.fit([x1, x2], y, epochs=500, batch_size=10,validation_split=0.3)

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

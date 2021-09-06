import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model



#tf.compat.v1.disable_eager_execution()

session = tf.compat.v1.InteractiveSession()

def lee_mdoel(trainX, trainY, testX, testY):
    # print(trainX.shape,trainY.shape,testX.shape,testY.shape)


    input = tf.keras.Input(shape=(102, 192, 1))
    con1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(input)
    max1 = keras.layers.MaxPool2D(pool_size=2, padding='same')(con1)
    con2 = keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(max1)
    max2 = keras.layers.MaxPool2D(pool_size=2, padding='same')(con2)
    con3 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(max2)
    max3 = keras.layers.MaxPool2D(pool_size=2, padding='same')(con3)
    con4 = keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(max3)
    max4 = keras.layers.MaxPool2D(pool_size=2, padding='same')(con4)
    r = keras.layers.Reshape((-1, 224))(max4)
    print(r.shape,max4.shape)
    lstm1 = keras.layers.LSTM(units=128, return_sequences=True)(r)
    lstm1=keras.layers.Dropout(0.2)(lstm1)
    lstm2 = keras.layers.LSTM(units=128, return_sequences=False)(lstm1)
    lstm2=keras.layers.Dropout(0.2)(lstm2)
    out = keras.layers.Dense(31, activation='softmax')(lstm2)
    # out=keras.layers.Dropout(0.2)(out)
    out = tf.expand_dims(out, 0)
    out = tf.keras.backend.permute_dimensions(out, (2, 1, 0))
    out = keras.layers.Dense(102)(out)
    # out = tf.tile(tf.expand_dims(out, 0), [102, 1, 1])
    out = tf.keras.backend.permute_dimensions(out, (1, 2, 0))

    model = Model(inputs=input, outputs=out)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
                  loss=tf.keras.losses.MSE, metrics=['accuracy'])
    model.fit(x=trainX, y=trainY, batch_size=128, epochs=100)
    loss, acc = model.evaluate(testX, testY, batch_size=32)
    print(loss, acc)
    predictY = model.predict(testX, batch_size=32)
    # print(predictY)


def Cov1_model(trainX,trainY,testX,testY):
    input=tf.keras.Input(shape=(102,192))

    # input1=tf.keras.layers.Flatten()(input)
    #
    # input2=tf.keras.backend.expand_dims((input1),axis=2)
    #print(input2.shape)
    #keras.backend.permute_dimensions(input2,(0,2,1))
    cov1=keras.layers.Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(input)
    max1=-keras.layers.MaxPool1D(pool_size=2,padding='same')(cov1)
    cov2=keras.layers.Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(max1)
    max2=keras.layers.MaxPool1D(pool_size=2,padding='same')(cov2)
    cov3 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(max2)
    max3 = keras.layers.MaxPool1D(pool_size=2, padding='same')(cov3)
    cov4 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(max3)
    max4 = keras.layers.MaxPool1D(pool_size=2, padding='same')(cov4)

    # r=keras.layers.Flatten()(cov4)

    lstm1=keras.layers.LSTM(units=128,return_sequences=True)(max4)
    lstm2 = keras.layers.LSTM(units=128, return_sequences=False)(lstm1)
    #d1=keras.layers.Dense(102)
    #out = keras.layers.Dense(31, activation='softmax')(lstm2)
    # out=keras.layers.Dropout(0.2)(out)

    out=tf.expand_dims(lstm2, 0)
    out=tf.keras.backend.permute_dimensions(out, (2, 1, 0))
    out=keras.layers.Dense(102)(out)
    # out = tf.tile(tf.expand_dims(out, 0), [102, 1, 1])
    out=tf.keras.backend.permute_dimensions(out, (1, 2, 0))
    out = keras.layers.Dense(31, activation='softmax')(out)
    model = Model(inputs=input, outputs=out)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.MSE, metrics=['accuracy'])
    model.fit(x=trainX, y=trainY, batch_size=32, epochs=1000)
    loss, acc = model.evaluate(testX, testY, batch_size=64)
    print(loss, acc)


def RNN_model(trainX,trainY,testX,testY):
    input=tf.keras.Input(shape=(102,192))
    d1=keras.layers.Dense(512)(input)

    out = keras.layers.Dense(31, activation='softmax')(d1)
    # out=keras.layers.Dropout(0.2)(out)
    # out = tf.expand_dims(out, 0)
    #print(out.shape)
    #out = tf.keras.backend.permute_dimensions(out, (2, 1, 0))
    #out = keras.layers.Dense(102)(out)
    # out = tf.tile(tf.expand_dims(out, 0), [102, 1, 1])
    #out=tf.expand_dims(out,0)
    #out = tf.keras.backend.permute_dimensions(out, (1, 2, 0))
    model = Model(inputs=input, outputs=out)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x=trainX, y=trainY, batch_size=64, epochs=100,steps_per_epoch=5000)
    loss, acc = model.evaluate(testX, testY, batch_size=256)
    print(loss, acc)


def generate_batch_data_random(x, y, batch_size):
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
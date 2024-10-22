import tensorflow as tf
import tensorflow.python.keras.backend as backend
import keras 
from keras import models, layers, callbacks
import numpy as np
import os
import random
import cv2
import time


def get_session(gpu_fraction=0.9):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


model = models.Sequential()
model.add(layers.Conv2D(32, (7, 7), padding='same',
                 input_shape=(176, 200, 1),
                 activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(13, activation='softmax'))

learning_rate = 0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate)#, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = callbacks.TensorBoard(log_dir="logs/STAGE2-{}-{}".format(int(time.time()), learning_rate))

train_data_dir = "train_data"

def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


hm_epochs = 5

for i in range(hm_epochs):
    current = 0
    increment = 11
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        if current + increment + 1> maximum:
            not_maximum = False
        try:
            print("WORKING ON {}:{}, EPOCH:{}".format(current, current+increment, i))

            choices = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: [],
                       5: [],
                       6: [],
                       7: [],
                       8: [],
                       9: [],
                       10: [],
                       11: [],
                       12: []
                       }

            for file in all_files[current:current+increment]:
                try:
                    full_path = os.path.join(train_data_dir, file)
                    data = np.load(full_path, allow_pickle=True)
                    data = list(data)
                    for d in data:
                        choice = np.argmax(d[0])
                        choices[choice].append([d[0], d[1]])
                except Exception as e:
                    print(str(e))

            lengths = check_data(choices)

            lowest_data = min(lengths)

            for choice in choices:
                random.shuffle(choices[choice])
                choices[choice] = choices[choice][:lowest_data]

            check_data(choices)

            train_data = []

            for choice in choices:
                for d in choices[choice]:
                    train_data.append(d)

            random.shuffle(train_data)
            print(len(train_data))

            test_size = 100
            batch_size = 128  # 128 best so far.

            x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 1)
            y_train = np.array([i[0] for i in train_data[:-test_size]])

            x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 1)
            y_test = np.array([i[0] for i in train_data[-test_size:]])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      epochs=1,
                      verbose=1, callbacks=[tensorboard])

            model.save("GIGABOT_epoch_lr_{}_{}_V1.keras".format(hm_epochs, learning_rate))
        except Exception as e:
            print(str(e))
        current += increment
        if current > maximum:
            not_maximum = False
        


            
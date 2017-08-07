from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

batch_size = 64
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_train',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_valid',
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical')



def pair_generator(cur_generator, batch_size, train=True):
    cur_cnt = 0
    while True:
        if train and cur_cnt % 4 == 1:
            # provide same image
            x1, y1 = train_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = train_generator.next()
            # print(y1)
            # print(np.sort(np.argmax(y1, 1), 0))
            y1_labels = np.argmax(y1, 1)
            has_move = list()
            last_not_move = list()
            idx2 = [-1 for i in range(batch_size)]

            for i, label in enumerate(y1_labels):
                if i in has_move:
                    continue
                for j in range(i+1, batch_size):
                    if y1_labels[i] == y1_labels[j]:
                        idx2[i] = j
                        idx2[j] = i
                        has_move.append(i)
                        has_move.append(j)
                        break
                if idx2[i] == -1:
                    # same element not found and hasn't been moved
                    if len(last_not_move) == 0:
                        last_not_move.append(i)
                        idx2[i] = i
                    else:
                        idx2[i] = last_not_move[-1]
                        idx2[last_not_move[-1]] = i
                        del last_not_move[-1]
            x2 = list()
            y2 = list()
            for i2 in range(batch_size):
                x2.append(x1[idx2[i2]])
                y2.append(y1[idx2[i2]])
            # print(y2)
            x2 = np.asarray(x2)
            y2 = np.asarray(y2)
            # print(x2.shape)
            # print(y2.shape)
        else:
            x1, y1 = cur_generator.next()
            if y1.shape[0] != batch_size:
                x1, y1 = cur_generator.next()
            x2, y2 = cur_generator.next()
            if y2.shape[0] != batch_size:
                x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        one_hot_same = np.zeros([batch_size, 2])
        one_hot_same[np.arange(batch_size), same] = 1
        # print same
        # print one_hot_same
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        yield [x1, x2], [y1, y2, one_hot_same]


# load the dog model from file
model = load_model('dog_xception_tuned.h5')
# model = load_model('xception-tuned03-0.78.h5')

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate


def lr_decay(epoch):
    lrs = [0.0001, 0.0001, 0.00001, 0.000001, 0.000001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001]
    return lrs[epoch]

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss={'ctg_out_1': 'categorical_crossentropy',
                    'ctg_out_2': 'categorical_crossentropy',
                    'bin_out': 'binary_crossentropy'},
              loss_weights={
                    'ctg_out_1': 1.,
                    'ctg_out_2': 1.,
                    'bin_out': 0.5
              },
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
my_lr = LearningRateScheduler(lr_decay)
save_model = ModelCheckpoint('xception-tuned-cont{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5')
model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                    steps_per_epoch=16500/batch_size+1,
                    epochs=10,
                    validation_data=pair_generator(validation_generator, batch_size=batch_size),
                    validation_steps=20,
                    callbacks=[my_lr, save_model]) # otherwise the generator would loop indefinitely
model.save('dog_xception_tuned_cont.h5')



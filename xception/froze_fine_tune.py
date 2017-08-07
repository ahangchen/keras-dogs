import os

import keras
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.applications import Xception
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Lambda
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import plot_model

# set_session(tf.Session(config=config))
from keras.utils import plot_model

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=45,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 64
train_generator = train_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_train',
        # '/home/cwh/coding/data/cwh/test1',
        target_size=(299, 299),
        # batch_size=1,
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_valid',
        # '/home/cwh/coding/data/cwh/test1',
        target_size=(299, 299),
        # batch_size=1,
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


def eucl_dist(inputs):
    x, y = inputs
    return (x - y)**2


early_stopping = EarlyStopping(monitor='val_loss', patience=3)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
save_model = ModelCheckpoint('xception{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
if os.path.exists('dog_xception.h5'):
    model = load_model('dog_xception.h5')
else:
    # create the base pre-trained model
    input_tensor = Input(shape=(299, 299, 3))
    base_model = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    plot_model(base_model, to_file='xception_model.png')
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    base_model.output_layers = [base_model.layers[-1]]

    feature = base_model
    img1 = Input(shape=(299, 299, 3), name='img_1')
    img2 = Input(shape=(299, 299, 3), name='img_2')

    feature1 = feature(img1)
    feature2 = feature(img2)
    # let's add a fully-connected layer
    category_predict1 = Dense(100, activation='softmax', name='ctg_out_1')(
        Dropout(0.5)(feature1)
    )
    category_predict2 = Dense(100, activation='softmax', name='ctg_out_2')(
        Dropout(0.5)(feature2)
    )

    # concatenated = keras.layers.concatenate([feature1, feature2])
    dis = Lambda(eucl_dist, name='square')([feature1, feature2])
    # concatenated = Dropout(0.5)(concatenated)
    # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(concatenated)

    judge = Dense(2, activation='softmax', name='bin_out')(dis)
    model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])

    # model.save('dog_xception.h5')
    plot_model(model, to_file='model_combined.png')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='nadam',
                  loss={'ctg_out_1': 'categorical_crossentropy',
                        'ctg_out_2': 'categorical_crossentropy',
                        'bin_out': 'categorical_crossentropy'},
                  loss_weights={
                      'ctg_out_1': 1.,
                      'ctg_out_2': 1.,
                      'bin_out': 0
                  },
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                        steps_per_epoch=16500/batch_size+1,
                        epochs=30,
                        validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
                        validation_steps=1800/batch_size+1,
                        callbacks=[early_stopping, auto_lr, save_model])
    model.save('dog_xception.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

train_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_train',
        # '/home/cwh/coding/data/cwh/test1',
        target_size=(299, 299),
        # batch_size=1,
        batch_size=batch_size,
        class_mode='categorical')

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
cur_base_model = model.layers[2]
for layer in cur_base_model.layers[:105]:
    layer.trainable = False
for layer in cur_base_model.layers[105:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss={
                  'ctg_out_1': 'categorical_crossentropy',
                  'ctg_out_2': 'categorical_crossentropy',
                  'bin_out': 'categorical_crossentropy'},
              loss_weights={
                  'ctg_out_1': 1.,
                  'ctg_out_2': 1.,
                  'bin_out': 0.5
              },
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
save_model = ModelCheckpoint('xception-tuned-{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
model.fit_generator(pair_generator(train_generator, batch_size=batch_size),
                    steps_per_epoch=16500/batch_size+1,
                    epochs=30,
                    validation_data=pair_generator(validation_generator, train=False, batch_size=batch_size),
                    validation_steps=1800/batch_size+1,
                    callbacks=[auto_lr, save_model]) # otherwise the generator would loop indefinitely
model.save('dog_xception_tuned.h5')

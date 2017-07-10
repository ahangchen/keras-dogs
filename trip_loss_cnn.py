import keras
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np

from util import make_parallel

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        rotation_range=90,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_train',
        # '/hdd/cwh/test1',
        target_size=(224, 224),
        # batch_size=1,
        batch_size=128,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_valid',
        # '/hdd/cwh/test1',
        target_size=(224, 224),
        # batch_size=1,
        batch_size=128,
        class_mode='categorical')


def double_generator(cur_generator):
    while True:
        x1, y1 = cur_generator.next()
        if y1.shape[0] != 128:
            x1, y1 = cur_generator.next()
        x2, y2 = cur_generator.next()
        if y2.shape[0] != 128:
            x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        # print(y1.shape)
        # print(y2.shape)
        # print(same.shape)
        yield [x1, x2], [y1, y2, same]


# create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D(name='ave_pool')(x)

feature = Model(inputs=base_model.input, outputs=x)
img1 = Input(shape=(224, 224, 3), name='img_1')
img2 = Input(shape=(224, 224, 3), name='img_2')

feature1 = feature(img1)
feature2 = feature(img2)
# let's add a fully-connected layer
category_predict1 = Dense(100, activation='softmax', name='ctg_out_1')(
    Dropout(0.5)(
        Dense(1024, activation='relu')(
            feature1
        )
    )
)
category_predict2 = Dense(100, activation='softmax', name='ctg_out_2')(
    Dropout(0.5)(
        Dense(1024, activation='relu')(
            feature2
        )
    )
)


concatenated = keras.layers.concatenate([feature1, feature2])

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(concatenated)
judge = Dense(1, activation='sigmoid', name='bin_out')(x)

model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])


plot_model(model, to_file='model_2.png')
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam',
              loss={'ctg_out_1': 'categorical_crossentropy',
                    'ctg_out_2': 'categorical_crossentropy',
                    'bin_out': 'binary_crossentropy'},
              metrics=['accuracy'])
# model = make_parallel(model, 3)
# train the model on the new data for a few epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit_generator(double_generator(train_generator),
                    steps_per_epoch=200,
                    epochs=30,
                    validation_data=double_generator(validation_generator),
                    validation_steps=80,
                    callbacks=[early_stopping])
model.save('dog_inception.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in base_model.layers[:172]:
    layer.trainable = False
for layer in base_model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss={
                  'ctg_out_1': 'categorical_crossentropy',
                  'ctg_out_2': 'categorical_crossentropy',
                  'bin_out': 'binary_crossentropy'},
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit_generator(double_generator(train_generator),
                    steps_per_epoch=200,
                    epochs=60,
                    validation_data=double_generator(validation_generator),
                    validation_steps=80,
                    callbacks=[early_stopping]) # otherwise the generator would loop indefinitely
model.save('dog_inception_tuned.h5')

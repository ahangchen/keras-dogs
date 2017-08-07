import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras import Input
from keras.applications import Xception, InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, concatenate, maximum
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    width_shift_range=0.4,
    height_shift_range=0.4,
    rotation_range=90,
    zoom_range=0.7,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 48
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


def triple_generator(generator):
    while True:
        x, y = generator.next()
        yield x, [y, y, y, y]


early_stopping = EarlyStopping(monitor='val_loss', patience=3)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                            cooldown=0, min_lr=0)

if os.path.exists('dog_single_xception.h5'):
    model = load_model('dog_single_xception.h5')
else:
    # create the base pre-trained model
    input_tensor = Input(shape=(299, 299, 3))
    base_model1 = Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    base_model1 = Model(inputs=[base_model1.input], outputs=[base_model1.get_layer('avg_pool').output], name='xception')

    base_model2 = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    base_model2 = Model(inputs=[base_model2.input], outputs=[base_model2.get_layer('avg_pool').output],
                        name='inceptionv3')

    img1 = Input(shape=(299, 299, 3), name='img_1')

    feature1 = base_model1(img1)
    feature2 = base_model2(img1)

    # let's add a fully-connected layer
    category_predict1 = Dense(100, activation='softmax', name='ctg_out_1')(
        Dropout(0.5)(
            feature1
        )
    )

    category_predict2 = Dense(100, activation='softmax', name='ctg_out_2')(
        Dropout(0.5)(
            feature2
        )
    )

    category_predict = Dense(100, activation='softmax', name='ctg_out')(
        concatenate([feature1, feature2])
    )
    max_category_predict = maximum([category_predict1, category_predict2])

    model = Model(inputs=[img1], outputs=[category_predict1, category_predict2, category_predict, max_category_predict])

    # model.save('dog_xception.h5')
    plot_model(model, to_file='single_model.png')
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model1.layers:
        layer.trainable = False

    for layer in base_model2.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='nadam',
                  loss={
                      'ctg_out_1': 'categorical_crossentropy',
                      'ctg_out_2': 'categorical_crossentropy',
                      'ctg_out': 'categorical_crossentropy',
                      'maximum_1': 'categorical_crossentropy'
                  },
                  metrics=['accuracy'])
    # model = make_parallel(model, 3)
    # train the model on the new data for a few epochs

    model.fit_generator(triple_generator(train_generator),
                        steps_per_epoch=16500 / batch_size + 1,
                        epochs=30,
                        validation_data=triple_generator(validation_generator),
                        validation_steps=1800 / batch_size + 1,
                        callbacks=[early_stopping, auto_lr])
    model.save('dog_single_xception.h5')
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
cur_base_model = model.layers[1]
for layer in cur_base_model.layers[:105]:
    layer.trainable = False
for layer in cur_base_model.layers[105:]:
    layer.trainable = True

cur_base_model = model.layers[2]
for layer in cur_base_model.layers[:262]:
    layer.trainable = False
for layer in cur_base_model.layers[262:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss={
                      'ctg_out_1': 'categorical_crossentropy',
                      'ctg_out_2': 'categorical_crossentropy',
                      'ctg_out': 'categorical_crossentropy',
                      'maximum_1': 'categorical_crossentropy'
                  },
              metrics=['accuracy'])
batch_size = batch_size * 3 / 4
train_generator = test_datagen.flow_from_directory(
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

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
save_model = ModelCheckpoint('xception-tuned{epoch:02d}-{val_ctg_out_acc:.2f}.h5')
model.fit_generator(triple_generator(train_generator),
                    steps_per_epoch=16500 / batch_size + 1,
                    epochs=30,
                    validation_data=triple_generator(validation_generator),
                    validation_steps=1800 / batch_size + 1,
                    callbacks=[early_stopping, auto_lr, save_model])  # otherwise the generator would loop indefinitely
model.save('dog_single_xception_tuned.h5')

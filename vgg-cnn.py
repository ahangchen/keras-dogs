from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from util import make_parallel

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        rotation_range=60,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_train',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_valid',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

# create the base pre-trained model
input_tensor = Input(shape=(224, 224, 3))
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 100 classes
predictions = Dense(100, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# model = make_parallel(model, 3)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model on the new data for a few epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit_generator(train_generator,
                    steps_per_epoch=200,
                    epochs=30,
                    validation_data=validation_generator,
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
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit_generator(train_generator,
                    steps_per_epoch=200,
                    epochs=60,
                    validation_data=validation_generator,
                    validation_steps=80,
                    callbacks=[early_stopping]) # otherwise the generator would loop indefinitely
model.save('dog_inception_tuned.h5')

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

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
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/dog_keras_valid',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

# load the dog model from file
model = load_model('dog_inception_tuned.h5')


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
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
model.fit_generator(train_generator,
                    steps_per_epoch=200,
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=80,
                    callbacks=[early_stopping, auto_lr]) # otherwise the generator would loop indefinitely
model.save('dog_inception_tuned_cont.h5')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

batch_size = 96
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


# load the dog model from file
model = load_model('dog_single_xception_tuned.h5')
# model = load_model('xception-tuned03-0.78.h5')

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate


def lr_decay(epoch):
    lrs = [0.0001, 0.0001, 0.00001, 0.000001, 0.000001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001]
    return lrs[epoch]

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
              loss={'ctg_out_1': 'categorical_crossentropy'},
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
early_stopping = EarlyStopping(monitor='val_loss', patience=6)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
my_lr = LearningRateScheduler(lr_decay)
save_model = ModelCheckpoint('xception-tuned-cont{epoch:02d}-{val_acc:.2f}.h5')
model.fit_generator(train_generator,
                    steps_per_epoch=16500/batch_size+1,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=20,
                    callbacks=[my_lr, save_model]) # otherwise the generator would loop indefinitely
model.save('dog_xception_tuned_cont.h5')


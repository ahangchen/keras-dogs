from keras.backend import set_session
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import operator

from os import remove, path

from util import fwrite

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

model = load_model('dog_inception_tuned_cont.h5')
test_datagen = ImageDataGenerator()
valid_generator = test_datagen.flow_from_directory(
    '/hdd/cwh/dog_keras_valid',
    target_size=(224, 224),
    batch_size=128,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
        '/hdd/cwh/test',
        target_size=(224, 224),
        batch_size=128,
        shuffle=False,
        class_mode='categorical')
print test_generator.filenames

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
y = model.predict_generator(test_generator, 83, use_multiprocessing=True)
y_i = np.argmax(y, 1)
predict_path = 'predict.txt'
if path.exists(predict_path):
    remove(predict_path)
for i, idx in enumerate(y_i):
    fwrite(predict_path, str(label_idxs[idx][0]) + '\t' + test_generator.filenames[i][2:-4] + '\n')

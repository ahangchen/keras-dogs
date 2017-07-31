import shutil
import os

from keras.backend import set_session
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import operator

from os import remove, path

from os.path import exists

from util import fwrite

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
batch_size = 64
# model = load_model('xception/xception-tuned-cont09-0.82.h5')
model = load_model('xception/dog_xception_tuned_cont.h5')
# model = load_model('xception/xception-tuned-cont-froze-03-0.84.h5')
single_model = Model(inputs=model.layers[0].input, outputs=[model.layers[6].output])
model = single_model
# plot_model(model, to_file='single_model.png')
test_datagen = ImageDataGenerator(rescale=1./255,)
valid_generator = test_datagen.flow_from_directory(
    '/hdd/cwh/dog_keras_valid',
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

test_path = '/hdd/cwh/test'
label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
# print test_generator.filenameenames

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
y = model.predict_generator(test_generator, 10593/batch_size + 1, use_multiprocessing=True)
y_max_idx = np.argmax(y, 1)
y_max_es = np.max(y, 1)
predict_path = 'predict.txt'
if path.exists(predict_path):
    remove(predict_path)
#
# new_test_path = '/home/cwh/coding/data/cwh/test_p'
# if not os.path.exists(new_test_path):
#     os.makedirs(new_test_path)
for i, idx in enumerate(y_max_idx):
    fwrite(predict_path, str(label_idxs[idx][0]) + '\t' + test_generator.filenames[i][2:-4] + '\n')
    # if y_max_es[i] > 0.9:
    #     if not os.path.exists(new_test_path + '/' + str(label_idxs[idx][0])):
    #         os.makedirs(new_test_path + '/' + str(label_idxs[idx][0]))
    #     shutil.copy(test_path + '/' + test_generator.filenames[i],
    #                 new_test_path + '/' + str(label_idxs[idx][0]) + '/' + test_generator.filenames[i][2:])

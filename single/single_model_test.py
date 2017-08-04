import operator
from os import remove, path
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from util import fwrite

batch_size = 96
model = load_model('xception-tuned-cont04-0.79.h5')
plot_model(model, to_file='single_model.png')
test_datagen = ImageDataGenerator(rescale=1. / 255, )
valid_generator = test_datagen.flow_from_directory(
    '/hdd/cwh/dog_keras_valid',
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
    '/hdd/cwh/test',
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
# print test_generator.filenameenames

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
y = model.predict_generator(test_generator, 10593 / batch_size + 1, use_multiprocessing=True)
y_i = np.argmax(y, 1)
predict_path = 'predict.txt'
if path.exists(predict_path):
    remove(predict_path)
for i, idx in enumerate(y_i):
    fwrite(predict_path, str(label_idxs[idx][0]) + '\t' + test_generator.filenames[i][2:-4] + '\n')

from keras.backend import set_session
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import operator

from os import remove, path

from util import fwrite

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

model = load_model('dog_xception_tuned.h5')
# single_model = Model(inputs=model.layers[0].input, outputs=[model.layers[9].output])
# model = single_model
# plot_model(model, to_file='single_model.png')
test_datagen = ImageDataGenerator(rescale=1./255,)
valid_generator = test_datagen.flow_from_directory(
    '/home/cwh/coding/data/cwh/dog_keras_valid',
    target_size=(224, 224),
    batch_size=128,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
        '/home/cwh/coding/data/cwh/test',
        target_size=(224, 224),
        batch_size=128,
        shuffle=False,
        class_mode='categorical')
print test_generator.filenames


def double_generator(cur_generator, train=True):
    cur_cnt = 0
    while True:
        if train and cur_cnt % 4 == 1:
            # provide same image
            x1, y1 = test_generator.next()
            if y1.shape[0] != 128:
                x1, y1 = test_generator.next()
            # print(y1)
            # print(np.sort(np.argmax(y1, 1), 0))
            y1_labels = np.argmax(y1, 1)
            batch_size = y1_labels.shape[0]
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
            if y1.shape[0] != 128:
                x1, y1 = cur_generator.next()
            x2, y2 = cur_generator.next()
            if y2.shape[0] != 128:
                x2, y2 = cur_generator.next()
        same = (np.argmax(y1, 1) == np.argmax(y2, 1)).astype(int)
        # print(np.argmax(y1, 1))
        # print(np.argmax(y2, 1))
        # print(same)
        cur_cnt += 1
        yield [x1, x2], [y1, y2, same]


from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
[y1, y2, same] = model.predict_generator(double_generator(test_generator), 83, use_multiprocessing=True)
y1_i = np.argmax(y1, 1)
y2_i = np.argmax(y2, 1)
predict_path = 'two_predict.txt'
if path.exists(predict_path):
    remove(predict_path)
for i, idx in enumerate(y1_i):
    fwrite(predict_path, str(label_idxs[idx][0]) + '\t' + str(label_idxs[y2_i[i]][0]) + '\t' + str(same[i]) + '\n')

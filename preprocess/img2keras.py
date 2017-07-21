import os
from shutil import copy

label_cnt = 0
img_folder = '/home/cwh/coding/data/cwh/train2'
train_folder = '/home/cwh/coding/data/cwh/dog_keras_train'
valid_folder = '/home/cwh/coding/data/cwh/dog_keras_valid'
last_label = ' '
labels = list()
for i, file_name in enumerate(os.listdir('/home/cwh/coding/data/cwh/train2')):
    file_path = os.path.join(img_folder, file_name)
    if os.path.isfile(file_path):
        cur_label = file_name.split('_')[0]
        cur_label_length = len(cur_label)
        new_name = file_name[:cur_label_length] + os.sep + file_name
        if i % 10 == 0:
            new_path = os.path.join(valid_folder, new_name)
        else:
            new_path = os.path.join(train_folder, new_name)
        if cur_label not in labels:
            label_cnt += 1
            labels.append(cur_label)
            if not os.path.exists(os.path.join(img_folder, cur_label)):
                os.makedirs(os.path.join(img_folder, cur_label))
            if not os.path.exists(os.path.join(train_folder, cur_label)):
                os.makedirs(os.path.join(train_folder, cur_label))
            if not os.path.exists(os.path.join(valid_folder, cur_label)):
                os.makedirs(os.path.join(valid_folder, cur_label))
        copy(file_path, new_path)

print labels

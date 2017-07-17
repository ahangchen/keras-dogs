from keras import Input
from keras.engine import Model
from keras.layers import Dense
import numpy as np

words = Input(shape=[10], name='words')
encoder1 = Dense(30, activation='relu', name='encoder1')(words)
encoder2 = Dense(128, activation='relu', name='encoder2')(encoder1)
decoder1 = Dense(30, activation='relu', name='decoder1')(encoder2)
decoder2 = Dense(10, activation='sigmoid', name='decoder2')(encoder1)

model = Model(inputs=[words], outputs=[decoder2])
model.compile(optimizer='adam', loss={'decoder2': 'mse'})


def random_ints(batch_size=32):
    while True:
        x = np.random.randint(0, 2, [batch_size, 10])
        yield x, x
model.fit_generator(random_ints(batch_size=128), steps_per_epoch=200, epochs=20, validation_data=random_ints(128), validation_steps=20)

predict_input = np.random.randint(0, 2, [1, 10])
predict_output = model.predict(predict_input, 1)
print predict_input
print predict_output

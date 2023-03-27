from CNNModel import *
from Utils import *

import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


model = get_wavelet_cnn_model()
opt1 = tf.keras.optimizers.Adam(learning_rate=0.00002)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


path = 'D:\\Datasets_imp\\DDR_DR_grading\\'
x, y = load_data(path=path)

x = x.reshape(-1,150528)

# Oversampling
sampling_strategy = {1:6000, 2:6000, 3:6000, 4:6000}
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
x_s, y_s = ros.fit_resample(x, y)
x_s = x_s.reshape(-1,224,224,3)

x_train,x_test,y_train, y_test = train_test_split(x_s, y_s, test_size=0.20, random_state=42)

batch_size = 32
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = datagen.flow(x_train, y_train, batch_size = batch_size) 
steps_per_epoch = len(x_train) // batch_size

history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = 500,
        validation_data = (x_test, y_test))

predictions = model.predict(x_test)
pred = predictions.argmax(1)


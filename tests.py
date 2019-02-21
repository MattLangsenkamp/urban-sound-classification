import helper as h
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd


mfcc = 70

X = h.get_X_train(str(mfcc))
y, lenc, enc = h.get_y_train()
X_test = h.get_X_test(str(mfcc))

num_labels = y.shape[1]

filter_size = 2

model = Sequential()

model.add(Dense(256, input_shape = (mfcc,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X, y, batch_size=32, epochs=150, validation_split = 0.2)

predictions = model.predict(X_test)
predictions_labeled = h.recover_original_labels(predictions, enc, lenc)

h.save_cnn(model, 'mfcc_' + str(mfcc))

test = pd.read_csv('test/test.csv')
test['Class'] = predictions_labeled
test.to_csv('sub08.csv', index=False)
# pylint: disable=no-member
import h5py
from keras import utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers

if __name__ == "__main__":
    # Read data from hdf5 file
    # Data from independent .py file:
    # main_data_store.py
    hdf5_path = 'dataset.hdf5'
    hdf5_file = h5py.File(hdf5_path, "r")

    # Convert labels to One-Hot-Encoding
    encoder = LabelEncoder()
    encoder.fit(hdf5_file["train_labels"])
    encoded_train = encoder.transform(hdf5_file["train_labels"])
    encoded_test = encoder.transform(hdf5_file["test_labels"])

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_train = utils.to_categorical(encoded_train)
    dummy_test = utils.to_categorical(encoded_test)

    # Build NN Structure
    batch_size = 1280
    num_classes = dummy_train.shape[1]
    epochs = 3
    input_shape = (120,120, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation ='relu',
                    input_shape =input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adadelta(),
                metrics=['accuracy'])

    history = model.fit(hdf5_file["train_img"], dummy_train,
                     batch_size = batch_size,
                     epochs = epochs,
                     verbose = 2,
                     shuffle = 'batch'
                     )


    print(history)
   
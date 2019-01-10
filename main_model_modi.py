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
    encoder.fit(hdf5_file["labels"])
    encoded_data = encoder.transform(hdf5_file["labels"])

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = utils.to_categorical(encoded_data)[:,1:]
    

    # Build NN Structure
    batch_size = 64
    num_classes = dummy_y.shape[1]
    epochs = 20
    input_shape = (120,120, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation ='relu',
                    input_shape =input_shape,
                    padding='same',
                    dilation_rate=2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adadelta(),
                metrics=['accuracy'])
    
    history = model.fit(hdf5_file["img"], dummy_y,
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 2,
                        shuffle = 'batch'
                        )




   
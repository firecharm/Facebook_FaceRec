# pylint: disable=no-member
import h5py
from keras import utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping

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
    dummy_train = utils.to_categorical(encoded_train)[:,1:]
    # dummy_train = utils.to_categorical(encoded_train)
    dummy_test = utils.to_categorical(encoded_test)
    
    # Build NN Structure

    # Sets of hyper-parameters
    batch_sizes = [4,8,16,64]
    filters = [8,16,32,32]

    num_classes = dummy_train.shape[1]
    epochs = 20
    input_shape = (120,120, 3)

    # Store result
    results = []

    for batch_size,filter in zip(batch_sizes,filters):

        model = Sequential()
        model.add(Conv2D(filter, kernel_size=(3, 3),
                        activation ='relu',
                        input_shape =input_shape,
                        padding='same',
                        dilation_rate=2))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filter*2, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.Adadelta(),
                    metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor="val_loss",patience = 5, verbose = 1)

        model.fit(hdf5_file["train_img"], dummy_train,
                            validation_data=(hdf5_file["test_img"],dummy_test),
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose = 2,
                            shuffle = False,
                            callbacks=[early_stop]
                            )
        result = early_stop.best
        print("Best result with batch size", batch_size, "filter", filter)
        print("is",result)
        results.append(result)
    
    print("All Record:", results)




   
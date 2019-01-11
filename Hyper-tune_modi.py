# pylint: disable=no-member
import h5py
from keras import utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Read data from hdf5 file
        # Data from independent .py file:
        # main_data_store.py
    hdf5_path = 'dataset.hdf5'
    hdf5_file = h5py.File(hdf5_path, "r")

    X_train,X_test,y_train,y_test = train_test_split(hdf5_file["img"][...],hdf5_file['labels'][...],test_size=0.2)


    # Convert labels to One-Hot-Encoding
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_train = encoder.transform(y_train)
    encoded_test = encoder.transform(y_test)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_train = utils.to_categorical(encoded_train)[:,1:]
    dummy_test = utils.to_categorical(encoded_test)[:,1:]


        
    # Build NN Structure
    batch_sizes = [8,16,64]
    filters = [32,16,8]

    num_classes = 1083
    epochs = 40
    input_shape = (120,120, 3)

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
        
        early_stop = EarlyStopping(monitor="val_acc",patience = 5, verbose = 1)

        model.fit(X_train,dummy_train,
                            validation_data=(X_test,dummy_test),
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




   
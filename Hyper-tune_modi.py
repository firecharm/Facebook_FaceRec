# import talos
import talos as ta

# pylint: disable=no-member
import h5py
from keras import utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from talos.model.early_stopper import early_stopper

def CNN_model (x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Conv2D(params['first_neuron'], kernel_size=(3, 3),
                    activation ='relu',
                    input_shape =x_train.shape[1],
                    padding='same',
                    dilation_rate=2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params['first_neuron']*2, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adadelta(),
                metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size = params['batch_size'],
                        epochs = epochs,
                        verbose = 2,
                        validation_data=[x_val, y_val],
                        callbacks=early_stopper(5, mode='strict')
                        )

    return (history,model)


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
    
    epochs = 20
    

    p = {
     #number of neurons for the first layer
     'first_neuron':[4, 8, 16, 32],
     'batch_size': [4, 8, 16, 32]
     }
    
    # and run the scan
    h = ta.Scan(hdf5_file["img"], dummy_y,
                #calling parameter boundaries
                params=p,
                dataset_name='first_test',
                experiment_no='1',
                model=CNN_model,
                #grid_downsample determines what % of permutation will be randomly sampled and scanned
                #NOTE: repeat - 1% sample, see result, narrow the boundaries in the dictionary - until you get the result you want
                grid_downsample=0.1,
                val_split = 0.2)

    # use Scan object as input
    r = ta.Reporting('first_test_1.csv')
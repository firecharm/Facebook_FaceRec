
import numpy as np
import h5py

# Initialize hdf5 placeholder
# Need to create hdf5 placeholder for image (X) as numpy array, and its label (y)
# To initialize the placeholder, the shape of X is needed, this can be extracted from data_addrs
def Initialize_hdf5(train_addrs,test_addrs,shape0,shape1):
    hdf5_path = '/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/dataset.hdf5'

    # Initialize images (X)
    hdf5_file = h5py.File(hdf5_path, mode='w')

    train_shape = (len(train_addrs), shape0,shape1, 3)
    test_shape = (len(test_addrs), shape0,shape1, 3)   

    hdf5_file.create_dataset("train_img", train_shape, np.int8)
    hdf5_file.create_dataset("test_img", test_shape, np.int8)

    # Initialize label(y)
    dt = h5py.special_dtype(vlen=str) 
    hdf5_file.create_dataset("train_labels", (len(train_addrs),), dt)
    hdf5_file.create_dataset("test_labels", (len(test_addrs),), dt)
    
    # print ("Test database created properly:", hdf5_file['train_labels'][0])
    hdf5_file.close()


    # h = h5py.File(hd5file,'r')
    


# hdf5_file["train_labels"][...] = train_labels
# hdf5_file["test_labels"][...] = test_labels

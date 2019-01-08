# pylint: disable=no-member
import numpy as np
import h5py
import cv2

# Initialize hdf5 placeholder
# Need to create hdf5 placeholder for image (X) as numpy array, and its label (y)
# To initialize the placeholder, the shape of image is needed, this can be extracted from data_addrs
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
    
    # Fill in labels
    train_labels = [i.split('/')[-3] for i in train_addrs]
    test_labels = [i.split('/')[-3] for i in test_addrs]
    hdf5_file["train_labels"][...] = train_labels
    hdf5_file["test_labels"][...] = test_labels

    # Fill in images with Face Crop
    cascadePath = "/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/cascade/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)  

    for i in range(len(train_addrs)):
        # read an image and resize to (300,300)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (300,300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here

        face = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5)
        for (x,y,w,h) in face:
            image_face  = img[y:(y+h),x:(x+w)]
        image_face = cv2.resize(image_face,(shape0,shape1))

        hdf5_file["train_img"][i, ...] = image_face[None]

    for i in range(len(test_addrs)):
        # read an image and resize to (300,300)
        # cv2 load images as BGR, convert it to RGB
        addr = train_addrs[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (300,300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # add any image pre-processing here

        face = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5)
        for (x,y,w,h) in face:
            image_face  = img[y:(y+h),x:(x+w)]
        image_face = cv2.resize(image_face,(shape0,shape1))

        hdf5_file["test_img"][i, ...] = image_face[None]

    hdf5_file.close()

if __name__ == "__main__":
    # Get the train and test data address
    # Run independent .py file:
    # main_data_prep.py
    # files stored in txt file: Train_addrs.txt, Test_addrs.txt
    f = open('/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/Train_addrs.txt', 'r')
    train_addrs = f.read().splitlines()
    f.close()
    # 28380 frames, 946 person, 30 per person

    f = open('/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/Test_addrs.txt', 'r')
    test_addrs = f.read().splitlines()
    f.close()
    # 8726 frames, 946 person, max 10 frames per person

    # Initialize HDF5 database
    # Set 120 x 120 frame size for face input
    Initialize_hdf5(train_addrs,test_addrs,120,120)
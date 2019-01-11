# pylint: disable=no-member
import numpy as np
import h5py
import cv2
from Data_Prep import shrinkage

# Initialize hdf5 placeholder
# Need to create hdf5 placeholder for image (X) as numpy array, and its label (y)
# To initialize the placeholder, the shape of image is needed, this can be extracted from data_addrs
def Initialize_hdf5(data_addrs,shape0,shape1):
    hdf5_path = 'dataset.hdf5'

    # Initialize images (X)
    hdf5_file = h5py.File(hdf5_path, mode='w')

    data_shape = (len(data_addrs), shape0,shape1, 3)
    hdf5_file.create_dataset("img", data_shape, np.int8)
    
    # Initialize label(y)
    dt = h5py.special_dtype(vlen=str) 
    hdf5_file.create_dataset("labels", (len(data_addrs),), dt)
    
    # Fill in labels
    labels = [i.split('/')[-3] for i in data_addrs]
    hdf5_file["labels"][...] = labels
    

    # Fill in images with Face Crop
    cascadePath = "cascade/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)  

    for i in range(len(data_addrs)):
        # read an image and resize to (300,300)
        # cv2 load images as BGR, convert it to RGB
        addr = data_addrs[i]
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

        hdf5_file["img"][i, ...] = image_face[None]

    hdf5_file.close()

if __name__ == "__main__":
    # Get the data address
    # Run independent .py file:
    # Face_Crop_Cascade_Selection.py
    # files stored in txt file: Suceed_identify.txt
    f = open('Suceed_identify.txt', 'r')
    data_addrs = f.read().splitlines()
    data_addrs = shrinkage(data_addrs,50)
    f.close()
    # 358219 cleaned address, 1084 person
    # After shrinkage, 95,442 left

    # Initialize HDF5 database
    # Set 120 x 120 frame size for face input
    Initialize_hdf5(data_addrs,120,120)
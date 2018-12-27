from Data_Prep import data_clean,identity_name,train_test_split,shrinkage
from Data_Store import Initialize_hdf5
from Face_Crop import face_crop_to_array

if __name__ == "__main__":
    error_list = data_clean("/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/YTFErrors.csv")
    name_list = identity_name("/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/frame_images_DB", error_list)
    train_set,test_set = train_test_split(name_list)

    train_addrs = shrinkage(train_set,48) # take 48 frames per person for the training data
    test_addrs = shrinkage(test_set,10) # take 10 frames per person for the training data

    # Initialize HDF5 database
    Initialize_hdf5(train_addrs,test_addrs,120,120)

    # Crop the faces
    train_data = 0
    for file in train_addrs:
        face_img = face_crop_to_array(file,120,120)
        train_data += 1
    # print(train_data)
    # print(face_img)
    # print(hdf5_file["train_labels"][0])d
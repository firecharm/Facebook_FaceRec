from Data_Prep import data_clean,identity_name,train_test_split,shrinkage

if __name__ == "__main__":
    error_list = data_clean("YTFErrors.csv")
    name_list = identity_name("frame_images_DB", error_list)
    train_set,test_set = train_test_split(name_list)
    train_addrs = shrinkage(train_set,48) # take 48 frames per person for the training data
    test_addrs = shrinkage(test_set,10) # take 10 frames per person for the training data
    print(test_addrs[1])
# pylint: disable=no-member
from Data_Prep import train_test_split,shrinkage

if __name__ == "__main__":
    # Get the recognizeable frames
    # Cascade selection done in an independent .py file:
    # Face_Crop_Cascade_Selection
    # recogizeable frames stored in txt file: Suceed_identify.txt
    f = open('/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/Suceed_identify.txt', 'r')
    file_addrs = f.read().splitlines()
    f.close()
    # 358219 cleaned address, 1084 person

    train_addrs,test_addrs = train_test_split(file_addrs)
    
    train_addrs = shrinkage(train_addrs,30) # take 30 frames per person for the training data
    print(len(train_addrs))
    test_addrs = shrinkage(test_addrs,10) # take 10 frames per person for the training data
    print(len(test_addrs))

    # Save the cleaned result to disk
    with open('/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/Train_addrs.txt', 'w') as f:
        for item in train_addrs:
            f.write("%s\n" % item)
    
    with open('/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/Test_addrs.txt', 'w') as f:
        for item in test_addrs:
            f.write("%s\n" % item)
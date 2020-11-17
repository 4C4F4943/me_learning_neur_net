import numpy as np
import matplotlib.pyplot as plt 
import h5py

def load_dataset():
        train_dataset = h5py.File("train_catvnoncat.h5","r")
        train_x = np.array(train_dataset["train_set_x"][:])
        train_y = np.array(train_dataset["train_set_y"][:])

        test_dataset = h5py.File("test_catvnoncat.h5","r")

        test_x = np.array(test_dataset["test_set_x"][:])
        test_y = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        train_y = train_y.reshape((1,train_y.shape[0]))
        test_y = test_y.reshape((1,test_y.shape[0]))

        return train_x, train_y, test_x, test_y, classes

train_x, train_y, test_x, test_y, classes = load_dataset()

print("#"*10," the first reshape","#"*10)
print ("Train X shape: " + str(train_x.shape))
print ("Train Y shape: " + str(train_y.shape))
print ("Test X shape: " + str(test_x.shape))
print ("Test Y shape: " + str(test_y.shape))
x = int(input("enter in the index of the cat photo: "))
#0 no cat 1

index = x 
if x is not None:
        plt.imshow(train_x[index])
        plt.show()

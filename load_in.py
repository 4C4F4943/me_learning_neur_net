import numpy as np
import pickle
import os
import cv2
import math
import time
import matplotlib.pyplot as plt 

start = time.time()
################ load in youre optimized weights ###############
pickle_in = open("w.pickle","rb")
w = pickle.load(pickle_in)

########## define youre bias value ###################
b = np.float(-0.004794084986625586)
IMG_SIZE = 50

test_x = cv2.imread("test_cat_img.jpg",cv2.IMREAD_GRAYSCALE)
test_x = cv2.resize(test_x,(IMG_SIZE,IMG_SIZE))

######### to see the resized img #############
#plt.imshow(test_x)
#plt.show()

test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE,1)
test_x = test_x.reshape(test_x.shape[0], -1).T
test_x/255

print("test_x.shape: ",test_x.shape)
print("w.shape: ",w.shape)
print("b: ",b)

what_is = ["cat", "dog"]

def sigmoid(z):
    return 1/(1 + np.exp(-z))   

def predict(w, b, X):    
    # number of example
    m = X.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    print("this is the test: ",np.dot(w.T, X)+b)

    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):
        y_pred[0,i] = 1 if A[0,i] > 0.5 else 0 
        print("this is a inside: ",A[0,i])
        pass
    print("1y_pred.shape: ",y_pred.shape)
    return y_pred, A


y_pred,a = predict(w,b,test_x)
res = int(y_pred[0])
print("this is res : ",res)
print("it is a : {}\n".format(what_is[res]),"and this is A: ", a[0, 0])
end = time.time()
print("it did it in roughly {} seconds".format(round(end-start,3)))

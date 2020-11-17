import numpy as np
import math
import os
import cv2
import matplotlib.pyplot as plt
import random
import pickle
import time
DATADIR = "PetImages"
CATEGORIES = ["Cat","Dog"]
start = time.time()

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        break
    break

IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

train_data = []

def create_train_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        #0 is cat 1 is dog
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
"""
#################### this is if you would like to enter in the images youreselff ######################
create_train_data() --> it runs the function that turns the images into arrays
#print(len(train_data))
random.shuffle(train_data)

#plt.imshow(train_data[0], cmap="gray")
#plt.show()

X = []
Y = [[]]

for features, label in train_data:
    X.append(features)
    Y[0].append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)
################# here you can save the training data in a pickle to make the program run faster #############################
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
"""
########### load in the data ##############
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
X = X[:5000]
pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
Y = [Y[0][:5000]]

########## define the test set ##############
test_x = np.array(X[:1000])
test_y = np.array([Y[0][:1000]])


X = X.reshape(X.shape[0], -1).T
test_x = test_x.reshape(test_x.shape[0], -1).T

test_x = np.array(test_x)
test_y = np.array(test_y)

Y = np.array(Y)
print(Y.shape)
print(X.shape, test_x.shape)

print("the shape of X: ",X.shape)
print("the shape of Y: ",Y.shape)
print("the shape of test X: ",test_x.shape)
print("the shape of test Y: ",test_y.shape)
########### squish the data ################
X = X/255
test_x = test_x/255

####### this funtion should return a value between 0 and 1 to decide whether it is a cat or a dog ################
def sigmoid(z):
    #e = math.e
    #return 1/(1+e*np.exp(-z))
    return 1/(1+np.exp(-z))
    #return np.tanh(z)

######## define some random values for the weights #############
def initialize_parameters(dim):
    w = np.random.randn(dim, 1)*0.01
    b = 0
    return w, b
############# propagation refines the weights and biases ##############
def propagate(w, b, X, Y):

    m = X.shape[1]
    #calculate activation function/ this is the thing that gets the output of the neural network
    A = sigmoid((np.dot(w.T, X)+b))
    #gd = 1/m * np.dot(X,(A-Y).T)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    #find gradient (back propagation) / calculating the weights and biases
    
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)

    grads = {"dw": dw,"db": db} 
    return grads, cost
############ some more optimizing the weights and bias #################
def gradient_descent(w, b, X, Y, iterations, learning_rate):
    costs = []
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        
        #update parameters
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        costs.append(cost)
        if i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,"b": b}   
    #plt.plot(params["w"][:10]) 
    return params, costs
############## the function that predicts if it is a cat or a dog ##############
def predict(w, b, X):    
    # number of example
    m = X.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        y_pred[0,i] = 1 if A[0,i] >0.5 else 0 
        #print("the a thingy: " ,A[0,i])
        pass
    print("1y_pred.shape: ",y_pred.shape)
    return y_pred

############ where it all comes together ####################
def model(train_x, train_y, test_x, test_y, iterations, learning_rate):
    #print("the shape of x[0]: ",train_x.shape[0])
    w, b = initialize_parameters(train_x.shape[0])
    parameters, costs = gradient_descent(w, b, train_x, train_y, iterations, learning_rate)
    w = parameters["w"]
    b = parameters["b"]
    ######## you can write the bias to a file if you would like ############
    #bias_file = open("bias.data","w+")
    #print("this is b",b)
    #bias_file.write(str(b))

    pickle_out = open("w.pickle","wb")
    pickle.dump(w, pickle_out)
    pickle_out.close()

    #call the training function
    train_pred_y = predict(w, b, train_x)
    test_pred_y = predict(w, b, test_x)
    
    #print("these are the test predictions: ",test_pred_y[:10],"\n")
    for i in range(len(test_pred_y)):
        if test_pred_y[0][i] == 1. :
            print("it is a dog!")
        else:
            print("it is a cat!")
    print("these are the train predictions: ",train_pred_y[:10],"\n")
    print("2test Y_pred.shape: ", test_pred_y.shape)
    print("2test test_Y.shape: ", test_y.shape)

    print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
    print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))
    how_many =0
    for i in range(len(train_pred_y[0])):
        if train_pred_y[0][i] == 1.0:
            how_many+=1
            #print("it's a cat!",how_many)
    return costs

costs = model(X,Y,test_x,test_y,iterations=1000, learning_rate=0.0160)
end = time.time()
print("it did it roughly in {} minutes".format(round((end-start)/60,2)))

#plt.plot(costs)
#plt.xlabel('iterations')
#plt.ylabel('costs')
#plt.show()

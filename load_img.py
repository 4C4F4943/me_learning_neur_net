import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from rich.progress import track
import pickle
DATADIR = "PetImages"
CATEGORIES = ["Cat","Dog","Person"]

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
        for img in track(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
#create_train_data()
#print(len(train_data))
#random.shuffle(train_data)
"""
the shape of X:  (2500, 227547)
the shape of Y:  (1, 227547)
the shape of test X:  (2500, 1)
the shape of test Y:  (1, 1)"""
X = []
Y = [[]]

for features, label in train_data:
    X.append(features)
    Y[0].append(label)

#X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)

#pickle_out = open("Y.pickle","wb")
#pickle.dump(Y, pickle_out)
#pickle_out.close()

#pickle_out = open("X.pickle","wb")
#pickle.dump(X, pickle_out)
#pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("Y.pickle","rb")
Y = pickle.load(pickle_in)
test_x = np.array(X[:200])
test_y = np.array([Y[0][:200]])
#test_x = cv2.imread(os.path.join("test_dog_img.jpeg"),cv2.IMREAD_GRAYSCALE)
#test_x = cv2.resize(test_x, (IMG_SIZE,IMG_SIZE))
#test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE,1)

#test_y = np.array([[1]])
X = X.reshape(X.shape[0], -1).T
test_x = test_x.reshape(test_x.shape[0], -1).T
#test_x = np.array(test_x)
#test_y = np.array(test_y)
Y = np.array(Y)
print(Y.shape)
print(X.shape, test_x.shape)

print("the shape of X: ",X.shape)
print("the shape of Y: ",Y.shape)
print("the shape of test X: ",test_x.shape)
print("the shape of test Y: ",test_y.shape)
X = X/5555
test_x = test_x/5555
def sigmoid(z):
    return 1/(1+np.exp(-z))


def initialize_parameters(dim):
    w = np.random.randn(dim, 1)*0.01
    b = 0
    return w, b
#propagation refines the weights and biases 
def propagate(w, b, X, Y):
    #print("the shape of w: ",w.shape)
    #print("the shape of X: ",X.shape)
    #print("the shape of Y: ",Y.shape)

    m = X.shape[1]
    #print(m)
    #calculate activation function/ this is the thing that gets the output damn why the fuck do they call it that stupid stupid thing
    #print(X[0])
    A = sigmoid(np.dot(w.T, X)+b)
    #print("this si a: ",A)
    #print("the log: ",np.log(1 - A))
    #print("the log: ",(1 - A))
    #find the cY.shapeost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    #find gradient (back propagation) / calculating the weights and biases
    #print("np.dot: ",np.dot(X, (A-Y).T))
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    #print("the dw in propagate: ",dw.shape)
    cost = np.squeeze(cost)
    grads = {"dw": dw,"db": db} 
    #print("this is grads : ",grads["dw"])
    #print("this is cost: ",cost)
    return grads, cost

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

def predict(w, b, X):    
    # number of example
    m = X.shape[1]
    y_pred = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X)+b)
    
    for i in track(range(A.shape[1])):
        if A[0,i] > 0.3333:
            y_pred[0,i] = 1
        elif A[0,i] > 0.666666:
            y_pred[0,i] =2
        else:
            y_pred[0,i] = 0
        pass
    print("1y_pred.shape: ",y_pred.shape)
    return y_pred

def model(train_x, train_y, test_x, test_y, iterations, learning_rate):
    #print("the shape of x[0]: ",train_x.shape[0])
    w, b = initialize_parameters(train_x.shape[0])
    parameters, costs = gradient_descent(w, b, train_x, train_y, iterations, learning_rate)
    w = parameters["w"]
    b = parameters["b"]
    # predict 
    train_pred_y = predict(w, b, train_x)
    test_pred_y = predict(w, b, test_x)
    print("these are the test predictions: ",test_pred_y[:10],"\n")
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
#costs = model(train_x, train_y, test_x, test_y, iterations = 4000, learning_rate = 0.005)
costs = model(X,Y,test_x,test_y,iterations=1000, learning_rate=0.0005)
#plt.plot(costs)
#plt.xlabel('iterations')
#plt.ylabel('costs')

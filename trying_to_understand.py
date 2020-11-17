import matplotlib.pyplot as plt 
from statistics import mean
import numpy as np
import random
import time


######################## this is just a program where i learned to understand it better ##############################
start = time.time()
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#initialize youre weights and biases
def init_thing(one_thing):
    w = np.random.randn(one_thing,1) * 0.01
    b = 0
    return w,b

############# just makes random data ###################
def make_data():
    train_x = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    train_y = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for j in range(len(train_x)):
        for i in range(20):
            train_x[j].append(np.random.rand())
            train_y[j].append(np.random.rand())
    return np.array(train_x), np.array(train_y)

def propagate(w, b, X, Y):
    m = X.shape[1]
    #calculate activation function/ this is the thing that gets the output damn why the fuck do they call it that stupid stupid thing
    A = sigmoid(np.dot(w.T, X)+b)
    #print("this is a : ",A)
    #find the cost
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  
    #find gradient (back propagation) / calculating the weights and biases

    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    #print("the dw in propagate: ",dw.shape)
    cost = np.squeeze(cost)
    grads = {"dw": dw,"db": db} 
    #print("this is grads : ",grads["dw"])
    #print("this is cost: ",cost)
    return grads, cost

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m,b

def gradient_descent(w,b,X,Y,iterations, learning_rate):
    costs = []
    for i in range(iterations):
        #print("the shape of the weights inside the gradient_d: ", w.shape)
        grads,cost = propagate(w,b,X,Y)
        #optimizing the weights and biasees
        #w = w - learning_rate * grads["dw"]
        #b = b - learning_rate * grads["db"]
        costs.append(cost)
        if i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w":w, "b":b}
    return params, costs

############# it can't really predict anything ##################
def predict(w,b,X,m):
    #m = X.shape(1)
    y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        y_pred[0,i] = 1 if A[0,i] > 0.5 else 0
        pass
    return y_pred

def model(train_x,train_y, test_x,test_y,iterations, learning_rate):
    #init the weights and biasees
    print("the shape of tran_x: ",train_x.shape)
    w,b = init_thing(train_x.shape[0])
    print("int the model weights: ",w.shape)
    print("the shape of the weights: ",w.shape)
    params, costs = gradient_descent(w,b,train_x,train_y,iterations, learning_rate)
    m = train_x.shape[1]
    w = params["w"]
    
    b = params["b"]

    random.shuffle(train_x)
    random.shuffle(train_y)
    train_pred_y = predict(w,b,train_x,m)
    test_pred_y = predict(w,b,test_x,m)
    print("these are the test predictions: ",test_pred_y,"\n")
    print("these are the train predictions: ",train_pred_y,"\n")
    print("Train Acc: {} %".format(100 - np.mean(np.abs(train_pred_y - train_y)) * 100))
    print("Test Acc: {} %".format(100 - np.mean(np.abs(test_pred_y - test_y)) * 100))
    return costs


train_x,train_y = make_data()
test_x,test_y = make_data()
print(train_x.shape[1])
costs = model(train_x,train_y, test_x,test_y, iterations=3000, learning_rate=0.005)
end = time.time()
print("it took this long: ", end-start,"s")
"""
for i in range(len(train_x)):
    test = NN(train_x[i],train_y[i],w1,w2,b)
    if test > 0.5:
        print("it would be true if this was maybe something better: ",test)
    else:
        print("it would be nothing but who would ever know")"""
plt.show()
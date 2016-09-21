from keras.models import Sequential
from keras.layers import Dense
import sklearn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

class linearregression():
    def __init__(self, *args, **kwargs):
        self.author="Benon Mburu"
        self.load_dataset()
    def seeder_method(self):
        np.random.seed(0)
        X,y=sklearn.datasets.make_moons(200, noise=0.20)
        plt.scatter(X[:,0],X[:,1],s=14, c=y, cmap=plt.cm.Spectral)
        plt.show()
    def load_dataset(self):
        seed=7
        np.random.seed(seed)
        dataset=np.loadtxt("games.csv", delimiter=",")
        X = dataset[:,0:8]
        Y = dataset[:,8]
        model=Sequential()
            #fuuly conected layers are defined using Dense class
            #first argument specifies number of neurons
            #second argument specify the initialization method
            #third argument specifies activation function
        model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
        model.add(Dense(8, init='uniform', activation='relu'))
        model.add(Dense(1, init='uniform', activation='sigmoid'))
        #compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

app =linearregression()


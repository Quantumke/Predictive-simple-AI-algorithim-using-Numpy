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

app =linearregression()


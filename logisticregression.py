import numpy as np
from sklearn import *
import sklearn
class LogisticRegresion():
    def __init__(self, *args, **kwargs):
        self.author="Benson Mburu"

    def generate_data_set(self):
        np.random.seed(0)
        X,y=sklearn.dataset.make_moons(200, noise=0.20)
        plt.scatter(X[:,0],X[:,1],s=40, c=y,cmap=plt.cm.Spectral)

app=LogisticRegresion()

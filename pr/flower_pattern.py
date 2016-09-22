import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class machinelearning():

    def __init__(self, *args, **kwargs):
        self.author="Benson Nguru"
        self._hack_()

    def _hack_(self):
        #dataset=np.genfromtxt("petals.csv", delimiter=",")
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset=pandas.read_csv("petals.csv", names=names)
        dataset=dataset.reset_index()
        #dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        #dataset.hist()
        #scatter_matrix(dataset)
        #plt.show()
        #print (dataset.groupby('class').size())
        array = dataset.values
        X = array[:,0:4]
        Y = array[:,4]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)
        num_folds = 10
        num_instances = len(X_train)
        seed = 7
        scoring = 'accuracy'
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
            cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)




app=machinelearning()

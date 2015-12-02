"""
The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as confmat

class MetricsGenerator:
    def __init__(self, ytrue, ypred):
        self.ytrue = ytrue
        self.ypred = ypred
        self.cm = confmat(ytrue, ypred)
        self.classes = set(ytrue)

    def class_precision(self, label):
        self.tp = reduce(lambda x,y: x+y, [1 for i in self.ytrue if i==label])
        self.fp = reduce(lambda x,y: x+y, [1 for i in self.ypred if i==label])    
        return self.tp/(self.tp+self.fp+.0)

    def class_recall(self, label):
        self.tp = reduce(lambda x,y: x+y, [1 for i in self.ytrue if i==label])
        self.fn = reduce(lambda x,y: x+y, [1 for i in self.ypred if i!=label])
        return self.tp/(self.tp+self.fn+.0)

    def class_fscore(self, label):
        num = 2*self.class_precision(label)*self.class_recall(label)
        den = self.class_precision(label)+self.class_recall(label)
        return num/(den+.0)

    def macro_precision(self):
        cumsum = 0
        for class_ in self.classes:
            cumsum += self.class_precision(class_)
        return cumsum/3.

    def macro_recall(self):
        cumsum = 0
        for class_ in self.classes:
            cumsum += self.class_recall(class_)
        return cumsum/3.

    def macro_fscore(self):
        cumsum = 0
        for class_ in self.classes:
            cumsum += self.class_fscore(class_)
        return cumsum/3.        

    def confusion_matrix(ytrue, ypred):
        return self.cm

    def plot_confusion_matrix(
            self, title='Confusion matrix', cmap=plt.cm.Blues
            ):
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, sorted(list(self.classes)), rotation=45)
        plt.yticks(tick_marks, sorted(list(self.classes)))
        plt.tight_layout()
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Prevista')
        plt.show()
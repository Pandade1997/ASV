import os
import numpy as np
import matplotlib.pyplot as plt

class PlotFigure(object):  

    def __init__(self, save_path):
        self.save_path = save_path
        self.subplot_num = 0
        self.data = None
        self.label = None
        
    def update(self, data, label, name='test.png'):  
        self.data = data
        self.label = label
        assert len(self.data) == len(self.label)
        self.subplot_num = len(self.label)
        for i in xrange(self.subplot_num):
            Fig = plt.figure()   
            Ax = Fig.add_subplot(111)            
            x = range(0, len(self.data[i]))
            y = self.data[i]
            Ax.plot(x, y, '-')
            Ax.set_ylabel(self.label[i])
            name = self.label[i]
            plt.savefig(os.path.join(self.save_path, name))
            plt.close()
        
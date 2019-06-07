
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras_custom import round_weights


class Visualizer(Callback):
    def __init__(self, N, layers, ifactor, integer_bits=1, fractional_bits=2):
        super(Callback, self).__init__()
        self.layers = layers
        self.pgapp = QtGui.QApplication([])
        self.pgwin = pg.GraphicsWindow(title="Training monitor")
        self.pgwin.resize(300 * len(N), 600)
        pg.setConfigOptions(antialias=True)
        panel1 = self.pgwin.addPlot(title="Accuracy")
        panel1.showGrid(x=True, y=True)
        panel2 = self.pgwin.addPlot(title="Loss")
        panel2.showGrid(x=True, y=True)
        self.plot1_train = panel1.plot(pen='y')
        self.plot1_val = panel1.plot(pen='g')
        self.plot2_train = panel2.plot(pen='y')
        self.plot2_val = panel2.plot(pen='g')
        self.pgwin.nextRow()
        panels_hist = [self.pgwin.addPlot(title="Weights L%s" % (i+1)) for i in xrange(len(N) - 1)]
        self.plots_hist1 = [p.plot(stepMode=True, fillLevel=0, brush='b') for p in panels_hist]
        self.plots_hist2 = [p.plot(stepMode=True, fillLevel=0, brush='y') for p in panels_hist]
        self.plotdat1_train = []
        self.plotdat1_val = []
        self.plotdat2_train = []
        self.plotdat2_val = []
        self.N = N
        self.ifactor = ifactor
        self.integer_bits = integer_bits
        self.fractional_bits = fractional_bits

    def _update(self, log):
        self.plotdat1_train.append(np.mean(self.acc_acc))
        self.plotdat1_val.append(log['val_acc'])
        self.plotdat2_train.append(np.mean(self.loss_acc))
        self.plotdat2_val.append(log['val_loss'])
        self.plot1_train.setData(self.plotdat1_train)
        self.plot1_val.setData(self.plotdat1_val)
        self.plot2_train.setData(self.plotdat2_train)
        self.plot2_val.setData(self.plotdat2_val)
        weights_raw = [l.get_weights() for l in self.layers]
        weights_round = [round_weights(w[0], self.integer_bits, self.fractional_bits, self.ifactor) for w in weights_raw]
        self.weights_round = weights_round
        for i in xrange(len(self.N) - 1):
            y,x = np.histogram(weights_raw[i][0].flatten(), bins=100)
            self.plots_hist1[i].setData(x, y)
            if len(weights_raw[i][0].shape) > 1:
                y,x = np.histogram(weights_round[i].flatten(), bins=100)
                self.plots_hist2[i].setData(x, y)
        self.pgapp.processEvents()

    def on_train_begin(self, logs={}):
        self.acc_acc = []
        self.loss_acc = []

    def on_batch_end(self, batch, logs={}):
        self.acc_acc.append(logs['acc'])
        self.loss_acc.append(logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        self._update(logs)
        self.acc_acc = []
        self.loss_acc = []

    def save(self):
        for i, w in enumerate(self.weights_round):
            np.savetxt('data/tmp/weights-round-l%s.dat' % i, w.flatten())
        np.savetxt('data/tmp/train_acc.dat', self.plotdat1_train)
        np.savetxt('data/tmp/valid_acc.dat', self.plotdat1_val)

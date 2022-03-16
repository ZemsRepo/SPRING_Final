import math
import copy

import numpy as np
from scipy import signal
import heapq as hq
from numba import cuda,jit


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from time import perf_counter


@jit(nopython=True)
def dist_func(x, y):
    return (x - y)**2

@jit(nopython=True)
def _updateStwm(querySequence,stwmD,stwmI,sampling,current_value):
    Q = querySequence
    D = stwmD
    I = stwmI
    D[:, 1:] = D[:, :-1]
    I[:, 1:] = I[:, :-1]
    m = len(D)
    N = sampling
    st = current_value

    D[0, 0] = dist_func(Q[0], st)
    I[0, 0] = N

    for i in range(1, m):
        D[i, 0] = dist_func(Q[i], st) + min(D[i - 1, 1], D[i, 1], D[i - 1, 0])

        Min = min(D[i - 1, 1], D[i, 1], D[i - 1, 0])
        if Min == D[i - 1, 1]:
            I[i, 0] = I[i - 1, 1]
        elif Min == D[i, 1]:
            I[i, 0] = I[i, 1]
        elif Min == D[i - 1, 0]:
            I[i, 0] = I[i - 1, 0]

    return D, I

class Signal():
    matchedSequenceCandidateArray = []
    matchedSequence = []
    stwmDCandidateArray = []
    dtwDistanceSequence = []

    def __init__(self, fullSequencePath = None, querySequencePath = None,threshold = 1):
        self.querySequence = np.loadtxt(querySequencePath)
        if len(self.querySequence)>100:
            self.querySequence = signal.savgol_filter(signal.resample(self.querySequence,100),2,1)
        self.fullSequence = np.loadtxt(fullSequencePath)
        self.querySequenceLength = len(self.querySequence)
        self.presentSequenceLength = self.querySequenceLength*15
        self.presentSequence = np.zeros(self.presentSequenceLength)
        self.stwmD = np.zeros([self.querySequenceLength,self.presentSequenceLength]);self.stwmD[:,0] = np.inf
        self.stwmI = np.zeros([self.querySequenceLength,self.presentSequenceLength])
        self.threshold = threshold

    def updateSequence(self):
        global N
        self.st = self.fullSequence[N]
        self.presentSequence[1:] = self.presentSequence[:-1]
        self.presentSequence[0] = self.st

    def updateStwm(self):
        global N
        self.stwmD,self.stwmI = _updateStwm(self.querySequence,self.stwmD,self.stwmI,N,self.st)

    def getMatchedSequence(self):
        global N
        D = self.stwmD
        I = self.stwmI

        if D[-1, 0] <= self.threshold:
            if D[-1, 1] > self.threshold:
                self.stwmDCandidateArray = []
                self.matchedSequenceCandidateArray = []
            self.stwmDCandidateArray.append(D[-1, 0])
            self.matchedSequence_cand = self.presentSequence[0:int(N - I[-1, 0]) + 1]
            self.matchedSequenceCandidateArray.append(copy.copy(self.matchedSequence_cand))

        if D[-1, 0] > self.threshold:
            if D[-1, 1] <= self.threshold:
                self.stwmDCandidateArray = self.stwmDCandidateArray[::-1]
                local_min_index = len(self.stwmDCandidateArray) - 1 - self.stwmDCandidateArray.index(min(self.stwmDCandidateArray[::-1]))
                self.matchedSequence = self.matchedSequenceCandidateArray[local_min_index]
                self.stwmDCandidateArray = []
                self.matchedSequenceCandidateArray = []
            if D[-1, 1] > self.threshold:
                self.stwmDCandidateArray = []
                self.matchedSequenceCandidateArray = []

    def setPlot1(self,title = None,pen = "white"):
        self.p1 = win.addPlot(title=title)
        self.curve1 = self.p1.plot(pen = pen)

    def setPlot2(self,title = None,pen = "white"):
        self.p2 = win.addPlot(title=title)
        self.curve2 = self.p2.plot(pen = pen)

    def updatePlot1(self):
        global N
        self.curve1.setData(self.presentSequence[::-1])
        self.curve1.setPos(N,0)

    def updatePlot2(self):
        self.curve2.setData(self.matchedSequence[::-1])

    def updateData(self):
        self.updateSequence()
        self.updateStwm()
        self.getMatchedSequence()


class Metric():

    def __init__(self,signal1,signal2):
        self.presentSequenceLength = len(signal1.presentSequence)
        self.presentSequence = np.zeros(self.presentSequenceLength)


    def updateSequence(self,signal1,signal2):
        self.st = signal1.st * signal2.st
        self.presentSequence[1:] = self.presentSequence[:-1]
        self.presentSequence[0] = self.st


    def setPlot1(self,title = None,pen = "white"):
        self.p1 = win.addPlot(title=title)
        self.curve1 = self.p1.plot(self.presentSequence[::-1], pen=pen)

    def updatePlot1(self):
        global N
        self.curve1.setData(self.presentSequence[::-1])
        self.curve1.setPos(N,0)






currentWithCmt = Signal(fullSequencePath="V2BCurrent_Segment.csv", querySequencePath="V2BCurrent_CMT.csv", threshold= 10000)
CurrentWithPuls = Signal(fullSequencePath="V2BCurrent_Segment.csv", querySequencePath="V2BCurrent_Puls.csv", threshold= 10000)
voltageWithZuendfehler = Signal(fullSequencePath="V2BVoltage_Segment.csv", querySequencePath="V2BVoltage_Zuendfehler.csv", threshold= 600)
voltageWithSpritzer = Signal(fullSequencePath="V2BVoltage_Segment.csv", querySequencePath="V2BVoltage_Spritzer5.csv", threshold= 200)
power = Metric(currentWithCmt,voltageWithZuendfehler)


app = pg.mkQApp("Spring Dashboard")
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle("Spring basic Dashboard")
win.resize(1000,600)

currentWithCmt.setPlot1("Strom Datastream",pen=(217,83,25))
currentWithCmt.setPlot2("CMT")
CurrentWithPuls.setPlot2("Puls")

win.nextRow()

voltageWithZuendfehler.setPlot1("Spannung Datastream",pen=(0,114,189))
voltageWithZuendfehler.setPlot2("Zündfehler")
voltageWithSpritzer.setPlot2("Spritzer")

win.nextRow()

power.setPlot1("Leistung Datastream",pen=(126,47,142))

qGraphicsGridLayout = win.ci.layout
qGraphicsGridLayout.setColumnStretchFactor(0,2)



def updateData():
    global N
    currentWithCmt.updateData()
    CurrentWithPuls.updateData()
    voltageWithZuendfehler.updateData()
    voltageWithSpritzer.updateData()
    power.updateSequence(currentWithCmt,voltageWithZuendfehler)

    currentWithCmt.updatePlot1()
    currentWithCmt.updatePlot2()
    CurrentWithPuls.updatePlot2()

    voltageWithZuendfehler.updatePlot1()
    voltageWithZuendfehler.updatePlot2()
    voltageWithSpritzer.updatePlot2()

    power.updatePlot1()

    N+=1


N = 0
timer = pg.QtCore.QTimer()
timer.timeout.connect(updateData)
timer.start(1)

if __name__ == '__main__':
    pg.exec()
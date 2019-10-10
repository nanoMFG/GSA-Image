from __future__ import division
import numpy as np
import scipy as sc
import cv2, sys, time, json, copy, subprocess, os
from skimage import transform
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from PIL import Image
import collections
from .gsaimage import FilterPattern, RemoveScale, Crop, DrawScale, InitialImage, Modification

class ImageEditor(QtGui.QWidget):
    submitClicked = QtCore.pyqtSignal(int,int,object) # sem_id, px_per_um, mask
    def __init__(self,sem_id,privileges=None,mode='local',parent=None):
        super(ImageEditor,self).__init__(parent=parent)
        self.img = np.zeros((1,1))
        self.sem_id = sem_id
        self.mode = mode

        self.nextButton = QtGui.QPushButton("Draw Scale")
        self.backButton = QtGui.QPushButton("Back")
        self.layerLabel = QtGui.QLabel('Initial Image')

        self.layer_edit = QtGui.QStackedWidget()

        self.imgItem = pg.ImageItem()
        self.imgBox = pg.GraphicsLayoutWidget()
        self.imgBox_VB = self.imgBox.addViewBox(row=1,col=1)
        self.imgBox_VB.addItem(self.imgItem)
        self.imgBox_VB.setAspectLocked(True)

        self.modifications = collections.OrderedDict()        
        self.modifications['Initial Image'] = InitialImage(
                                                img_item = self.imgItem,
                                                properties = {'mode':self.mode})
        self.modifications['Initial Image'].set_image(self.img)
        self.modifications['Draw Scale'] = DrawScale(
                                                self.modifications['Initial Image'],
                                                img_item = self.imgItem,
                                                properties=self.modifications['Initial Image'].properties)
        self.modifications['Remove Scale'] = RemoveScale(
                                                self.modifications['Draw Scale'],
                                                img_item = self.imgItem,
                                                properties = self.modifications['Draw Scale'].properties)
        self.modifications['Template Matching'] = FilterPattern(
                                                self.modifications['Remove Scale'],
                                                img_item = self.imgItem,
                                                properties = self.modifications['Remove Scale'].properties)
        self.modifications['Review'] = Review(
                                                self.modifications['Template Matching'],
                                                img_item = self.imgItem,
                                                properties = self.modifications['Template Matching'].properties)

        self.step_labels = list(self.modifications.keys())[1:]+['Submit']

        for key, value in self.modifications.items():
            self.layer_edit.addWidget(value)
        self.layer_edit.setCurrentWidget(self.modifications['Initial Image'])

        self.layout = QtGui.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(self.imgBox,0,0,3,1)
        self.layout.addWidget(self.layerLabel,0,1,1,2)
        self.layout.addWidget(self.layer_edit,1,1,1,2)
        self.layout.addWidget(self.backButton,2,1,1,1)
        self.layout.addWidget(self.nextButton,2,2,1,1)

        self.nextButton.clicked.connect(self.next)
        self.backButton.clicked.connect(self.back)

    def loadImage(self,data,thread_id,info):
        self._id = thread_id
        self.img = np.array(Image.open(io.BytesIO(data)))
        self.modifications['Initial Image'].set_image(self.img)

    def next(self):
        if self.img:
            index = self.layer_edit.currentIndex()
            if index < self.layer_edit.count() - 1:
                self.nextButton.setText(self.step_labels[index+1])

                self.layer_edit.setCurrentIndex(index+1)
                self.layer_edit.currentWidget().update_view()
            elif index == self.layer_edit.count() - 1:
                mask = self.modifications['Template Matching'].properties['total_mask']
                scale = self.modifications['Draw Scale'].properties['scale']
                px_per_um = int(1/scale)
                self.submitClicked.emit(self.sem_id,px_per_um,properties['mask'])

    def back(self):
        if self.img:
            index = self.layer_edit.currentIndex()
            if index > 0:
                self.nextButton.setText(self.step_labels[index-1])

                self.layer_edit.setCurrentIndex(index-1)
                self.layer_edit.currentWidget().update_view()


class Review(Modification):
    def __init__(self,mod_in=None,img_item=None,properties={}):
        super(Review,self).__init__(mod_in,img_item,properties)
        self.submitButton = QtGui.QPushButton('Submit')

        self.startImg = pg.ImageItem()
        self.startBox = pg.GraphicsLayoutWidget()
        self.startBox_VB = self.startBox.addViewBox(row=1,col=1)
        self.startBox_VB.addItem(self.startItem)
        self.startBox_VB.setAspectLocked(True)

        self.maskImg = pg.ImageItem()
        self.maskBox = pg.GraphicsLayoutWidget()
        self.maskBox_VB = self.maskBox.addViewBox(row=1,col=1)
        self.maskBox_VB.addItem(self.maskItem)
        self.maskBox_VB.setAspectLocked(True)

        self.px_per_um = QtGui.QLabel('')

        self.mainWidget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout(self.mainWidget)

        self.layout.addWidget(self.startBox,0,0,1,1)
        self.layout.addWidget(self.maskBox,0,1,1,1)
        self.layout.addWidget(QtGui.QLabel('Pixels/um:'),1,0,1,1)
        self.layout.addWidget(self.px_per_um,1,1,1,1)

    def update_image(self):
        properties = self.back_properties()
        self.px_per_um.setText(str(int(1/properties['scale'])))

        start_img = self.root().image()
        self.startImg.setImage(start_img,levels=(0,255))

        mask = np.array(properties['total_mask'])
        self.maskImg.setImage(start_img[mask],levels=(0,255))

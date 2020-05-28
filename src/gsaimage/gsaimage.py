from __future__ import division

import functools
import logging
import traceback
from collections import OrderedDict

import cv2
import json
import numpy as np
import os
import pyqtgraph as pg
import pyqtgraph.exporters
import scipy as sc
import subprocess
import sys
import time
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtGui, QtCore, QtWidgets
from pyqtgraph import Point
from skimage import transform
from skimage import util
from skimage.draw import circle as skcircle
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)
pg.setConfigOption('background', 'w')
pg.setConfigOption('imageAxisOrder', 'row-major')

tic = time.time()

def slow_update(func, pause=0.3):
    def wrapper(self):
        global tic
        toc = time.time()
        if toc - tic > pause:
            tic = toc
            return func(self)
        else:
            pass
    return wrapper

def mask_color_img(img, mask, color=[0, 0, 255], alpha=0.3):
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask] = color
    return cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

def check_extension(file_name, extensions):
    return any([(file_name[-4:]==ext and len(file_name)>4) for ext in extensions])

def errorCheck(success_text=None, error_text="Error!",logging=False,show_traceback=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
                if success_text:
                    success_dialog = QtGui.QMessageBox(self)
                    success_dialog.setText(success_text)
                    success_dialog.setWindowModality(QtCore.Qt.WindowModal)
                    success_dialog.exec()
            except Exception as e:
                error_dialog = QtGui.QMessageBox(self)
                error_dialog.setWindowModality(QtCore.Qt.WindowModal)
                error_dialog.setText(error_text)

                if logging:
                    logger.exception(traceback.format_exc())
                if show_traceback:
                    error_dialog.setInformativeText(traceback.format_exc())
                else:
                    error_dialog.setInformativeText(str(e))
                error_dialog.exec()

        return wrapper
    return decorator

class GSAImage(QtWidgets.QWidget):
    def __init__(self,mode='local',parent=None):
        super(GSAImage,self).__init__(parent=parent)
        self.mode = mode
        self.modifications = []
        self.selectedWidget = None

        if self.mode == 'nanohub':
            if 'TempGSA' not in os.listdir(os.getcwd()):
                os.mkdir('TempGSA')
            self.tempdir = os.path.join(os.getcwd(),'TempGSA')
            os.chdir(self.tempdir)

        self.mod_dict = {
        'Color Mask': ColorMask,
        'Canny Edge Detector': CannyEdgeDetection,
        'Dilate': Dilation,
        'Erode': Erosion,
        'Binary Mask': BinaryMask,
        # 'Find Contours': FindContours,
        'Filter Pattern': FilterPattern,
        'Blur': Blur,
        'Draw Scale': DrawScale,
        'Crop': Crop,
        'Domain Centers': DomainCenters,
        # 'Hough Transform': HoughTransform,
        'Erase': Erase,
        'Sobel Filter': SobelFilter,
        'Remove Scale': RemoveScale
        }

        self.wComboBox = pg.ComboBox()
        for item in sorted(list(self.mod_dict)):
            self.wComboBox.addItem(item)

        self.wOpenFileBtn = QtGui.QPushButton('Import')
        self.wOpenFileBtn.clicked.connect(self.importImage)

        self.wAddMod = QtGui.QPushButton('Add')
        self.wAddMod.clicked.connect(lambda: self.addMod(mod=None))

        self.wRemoveMod = QtGui.QPushButton('Remove')
        self.wRemoveMod.clicked.connect(self.removeMod)

        self.wExportImage = QtGui.QPushButton('Export')
        self.wExportImage.clicked.connect(self.exportImage)

        self.wExportState = QtGui.QPushButton('Export State')
        self.wExportState.clicked.connect(self.exportState)

        self.wImportState = QtGui.QPushButton('Import State')
        self.wImportState.clicked.connect(self.importState)

        self.wModList = QtGui.QListWidget()
        self.wModList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        # self.wModList.setIconSize(QtCore.QSize(72, 72))
        self.wModList.currentRowChanged.connect(self.selectMod)
        # self.wModList.currentRowChanged.connect(self.setListIcon)


        self.wMain = QtGui.QWidget()
        self.wMain.setFixedWidth(250)
        self.mainLayout = QtGui.QGridLayout()
        self.mainLayout.addWidget(self.wOpenFileBtn, 0,0)
        self.mainLayout.addWidget(self.wModList,2,0,1,2)
        self.mainLayout.addWidget(self.wAddMod, 3,0)
        self.mainLayout.addWidget(self.wRemoveMod,3,1)
        self.mainLayout.addWidget(self.wComboBox,4,0,1,2)
        self.mainLayout.addWidget(self.wExportImage,0,1)
        self.wMain.setLayout(self.mainLayout)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgItem = SmartImageItem()
        # self.wImgItem.sigImageChanged.connect(lambda: self.setListIcon(self.wModList.currentRow()))
        self.wImgBox_VB.addItem(self.wImgItem)
        self.wImgBox_VB.setAspectLocked(True)
        # self.wImgBox.setFixedWidth(400)

        self.wDetail = QtGui.QStackedWidget()
        # self.wDetail.setFixedWidth(400)

        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(self.wMain,0,0)
        self.layout.addWidget(self.wImgBox,0,1)
        self.layout.addWidget(self.wDetail,0,2)

    @classmethod
    def viewOnlyWidget(cls,d):
        obj = cls()
        cla = globals()[d['@class']]
        obj.modifications = cla.from_dict(d,obj.wImgItem).tolist()
        obj.layout.removeWidget(obj.wMain)
        obj.wMain.hide()
        obj.layout.removeWidget(obj.wDetail)
        obj.wDetail.hide()
        obj.layout.addWidget(obj.wModList,0,0)
        obj.layout.setColumnStretch(0,1)
        obj.layout.setColumnStretch(1,3)
        obj.updateAll()
        return obj

    def setListIcon(self,index=None):
        if isinstance(index,int) and index < self.wModList.count():
            pic = Image.fromarray(self.modifications[index].image())
            pic.thumbnail((72,72))
            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(ImageQt(pic)))
            self.wModList.item(index).setIcon(icon)

    def exportImage(self):
        if len(self.modifications) > 0:
            if self.mode == 'local':
                name = QtWidgets.QFileDialog.getSaveFileName(None, "Export Image", '', "All Files (*);;PNG File (*.png);;JPEG File (*.jpg)")[0]
                if name != '' and check_extension(name, [".png", ".jpg"]):
                    cv2.imwrite(name,self.modifications[-1].image())
            elif self.mode == 'nanohub':
                name = 'temp_%s.png'%int(time.time())
                cv2.imwrite(name,self.modifications[-1].image())
                subprocess.check_output('exportfile %s'%name,shell=True)
                # os.remove(name)
            else:
                return

    def exportState(self):
        if len(self.modifications) > 0:
            if self.mode == 'local':
                d = self.modifications[-1].to_dict()
                name = QtWidgets.QFileDialog.getSaveFileName(None, "Export Image", '', "All Files (*);;JSON File (*.json)")[0]
                if name != '' and check_extension(name, [".json"]):
                    with open(name,'w') as f:
                        json.dump(d,f)
            elif self.mode == 'nanohub':
                d = self.modifications[-1].to_dict()
                name = 'temp_%s.json'%int(time.time())
                with open(name,'w') as f:
                    json.dump(d,f)
                subprocess.check_output('exportfile %s'%name,shell=True)
                # os.remove(name)
            else:
                return

    def importState(self):
        if self.mode == 'local':
            try:
                file_path = QtGui.QFileDialog.getOpenFileName()
                if isinstance(file_path,tuple):
                    file_path = file_path[0]
                else:
                    return
                self.clear()
                with open(file_path,'r') as f:
                    state = json.load(f)
            except Exception as e:
                print(e)
                return
        elif self.mode == 'nanohub':
            try:
                file_path = subprocess.check_output('importfile',shell=True).strip().decode("utf-8")
                with open(file_path,'r') as f:
                    state = json.load(f)
                os.remove(file_path)
            except Exception as e:
                print(e)
                return
        else:
            return

        cla = globals()[state['@class']]
        self.modifications = cla.from_dict(state,self.wImgItem).tolist()
        self.updateAll()

    def manualImport(self,fpath):
        try:
            self.clear()
            img_fname = img_file_path.split('/')[-1]
            img_data = cv2.imread(img_file_path)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

            mod = InitialImage(img_item=self.wImgItem,properties={'mode':self.mode})
            mod.set_image(img_data)
            self.addMod(mod)
            self.w.setWindowTitle(img_fname)
        except:
            pass

    def importImage(self):
        if self.mode == 'local':
            try:
                img_file_path = QtGui.QFileDialog.getOpenFileName()
                if isinstance(img_file_path,tuple):
                    img_file_path = img_file_path[0]
                else:
                    return
                self.clear()
                img_fname = img_file_path.split('/')[-1]
                img_data = cv2.imread(img_file_path)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

                mod = InitialImage(img_item=self.wImgItem,properties={'mode':self.mode,"filename":img_fname})
                mod.set_image(img_data)
                self.addMod(mod)
                self.setWindowTitle(img_fname)
                mod.update_view()
            except Exception as e:
                print(e)
                return
        elif self.mode == 'nanohub':
            try:
                img_file_path = subprocess.check_output('importfile',shell=True).strip().decode("utf-8")
                img_fname = img_file_path.split('/')[-1]
                img_data = cv2.imread(img_file_path)
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)

                os.remove(img_file_path)
                self.clear()

                mod = InitialImage(img_item=self.wImgItem,properties={'mode':self.mode})
                mod.set_image(img_data)
                self.addMod(mod)
                self.setWindowTitle(img_fname)
                mod.update_view()
            except Exception as e:
                print(e)
                return
        else:
            return

    @errorCheck(logging=False,show_traceback=False)
    def updateAll(self):
        if len(self.modifications) == 0:
            self.clear()
        else:
            self.wModList.clear()
            while self.wDetail.count() > 0:
                self.wDetail.removeWidget(self.wDetail.widget(0))
            for i,mod in enumerate(self.modifications):
                self.wModList.addItem("%d %s"%(i,mod.name()))
                self.wDetail.addWidget(mod.widget())
            self.wModList.setCurrentRow(self.wModList.count()-1)

    @errorCheck(logging=False,show_traceback=False)
    def selectMod(self,index):
        print(index)
        if index >= 0:
            # try:
            self.modifications[index].update_view()
            # except:
                # pass
            if index == self.wDetail.count()-1:
                self.wDetail.setCurrentIndex(index)
        elif self.wModList.count() > 0:
            self.wModList.setCurrentRow(self.wModList.count()-1)

    def clear(self):
        self.wImgItem.clear()
        self.wModList.clear()
        self.modifications = []
        while self.wDetail.count() > 0:
            self.wDetail.removeWidget(self.wDetail.widget(0))

    def removeMod(self):
        if len(self.modifications) > 1:
            self.wDetail.removeWidget(self.modifications[-1].widget())
            try:
                self.modifications[-1].imageChanged.disconnect()
            except:
                pass
            del[self.modifications[-1]]
            self.wModList.takeItem(self.wModList.count()-1)
            if self.wModList.count() > 0:
                self.wModList.setCurrentRow(self.wModList.count()-1)

    @errorCheck(logging=False,show_traceback=True)
    def addMod(self,mod=None):
        if mod == None:
            if len(self.modifications) > 0:
                mod = self.mod_dict[self.wComboBox.value()](self.modifications[-1],self.wImgItem,properties={'mode':self.mode})
            else:
                return
        mod.imageChanged.connect(lambda img: self.wImgItem.setImage(img,levels=(0,255)))
        self.modifications.append(mod)
        self.wDetail.addWidget(mod.widget())
        self.wModList.addItem("%d %s"%(self.wModList.count(),mod.name()))
        if self.wModList.count() > 0:
            self.wModList.setCurrentRow(self.wModList.count()-1)

    def widget(self):
        return self

    def run(self):
        self.show()

class Modification(QtWidgets.QScrollArea):
    """
    Abstract class for defining modifications to an image. Modifications form a linked list with each object
    inheriting an input Modification. In this way, images are modified with sequential changes. Modification
    objects should also connect to GSAImage's ImageItem, which contains the main image display so that item
    can be updated.

    mod_in:         the Modification that the current object inherits.
    img_item:       GSAImage's ImageItem.
    properties:     a dictionary of properties that may be used for the modification.
    """

    imageChanged = QtCore.pyqtSignal(object)
    def __init__(self,mod_in=None,img_item=None,properties={},parent=None):
        super(Modification,self).__init__(parent=parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mod_in = mod_in
        self.img_item = img_item
        self.properties = properties
        if mod_in != None:
            self.img_out = self.mod_in.image()
        else:
            self.img_out = None

    def widget(self):
        """
        Returns the Modification's widget.
        """
        return self
    def image(self):
        """
        Returns the output image after modifications are applied.
        """
        return self.img_out.copy()
    def name(self):
        return 'Default Modification'
    def set_image(self,img):
        """
        Sets the output image manually. Only necessary for initializing.
        """
        self.img_out = img.astype(np.uint8)
    def update_image(self):
        """
        Abstract function for defining and applying modifications to the input image.
        """
        pass
    def update_view(self):
        """
        Updates the image display.
        """
        self.update_image()
        self.imageChanged.emit(self.img_out)
        self.img_item.setImage(self.img_out,levels=(0,255))
        return self.properties
    def delete_mod(self):
        """
        Deletes the modification and returns the inherited modification.
        """
        return self.mod_in
    def tolist(self):
        """
        Converts linked list to list.
        """
        if self.mod_in != None:
            return self.mod_in.tolist() + [self]
        else:
            return [self]
    def back_traverse(self,n):
        """
        Gives the modification that is n modifications back from the end of the list.

        n:              number of indices to traverse backwards on linked list.
        """
        if n != 0:
            if self.mod_in == None:
                raise IndexError('Index out of range (n = %d)'%n)
            elif n != 0:
                return self.mod_in.back_traverse(n-1)
        elif n == 0:
            return self
    def root(self):
        """
        Gives the first Modification in linked list.
        """
        if self.mod_in != None:
            return self.mod_in.root()
        else:
            return self
    def length(self):
        """
        Gives the length of the linked list.
        """
        if self.mod_in != None:
            return self.mod_in.length()+1
        else:
            return 1
    def back_properties(self):
        """
        Gives a dictionary containing all properties of current and inherited Modifications.
        """
        if self.mod_in != None:
            d = self.mod_in.back_properties()
            d.update(self.properties)
            return d
        else:
            d = {}
            d.update(self.properties)
            return d
    def to_dict(self):
        """
        Generic recursive function for converting object to json compatible storage object. Used for saving the state.
        """
        d = {}
        d['@module'] = self.__class__.__module__
        d['@class'] = self.__class__.__name__
        d['date'] = time.asctime()
        if self.mod_in != None:
            d['mod_in'] = self.mod_in.to_dict()
        else:
            d['mod_in'] = None
        d['properties'] = self.properties
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        """
        Generic recursive function for loading Modification using json compatible dictionary generated by to_dict().

        d:              dictionary from which to load.
        img_item:       GSAImage's ImageItem.
        """
        if d['mod_in'] != None:
            mod_in_dict = d['mod_in']
            mod_in_cls = globals()[mod_in_dict['@class']]
            mod_in = mod_in_cls.from_dict(mod_in_dict,img_item)
        else:
            mod_in = None
        return cls(mod_in,img_item,d['properties'])



class InitialImage(Modification):
    def name(self):
        return 'Initial Image'
    def to_dict(self):
        d = super(InitialImage,self).to_dict()
        d['img_out'] = self.img_out.tolist()
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(InitialImage,cls).from_dict(d,img_item)
        obj.set_image(np.asarray(d['img_out']))
        return obj

class RemoveScale(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(RemoveScale,self).__init__(mod_in,img_item,properties)

    def to_dict(self):
        d = super(RemoveScale,self).to_dict()
        d['scale_location'] = self.scale_location.currentText()

    def name(self):
        return 'Remove Scale'

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(RemoveScale,cls).from_dict(d,img_item)

        obj.scale_location.setIndex(obj.scale_location.findText(d['scale_location'],QtCore.Qt.MatchExactly))

        obj.update_image()
        obj.widget().hide()
        return obj

    def crop_image(self,img,box):
        pil_img = Image.fromarray(img)
        return np.array(pil_img.crop(box))

    def update_image(self,scale_location='Auto',tol=0.95):
        img_array = self.mod_in.image()
        img = Image.fromarray(img_array)
        width,height = img.size
        crop_img = None
        self.box = (0,0,width,height)
        if scale_location == 'Bottom' or scale_location == 'Auto':
            for row in range(height):
                if np.mean(img_array[row,:]==0)>tol:
                    self.box = (0,0,width,row)
                    crop_img = img.crop(self.box)
                    break
        elif scale_location == 'Top' or scale_location == 'Auto':
            for row in reversed(range(height)):
                if np.mean(img_array[row,:]==0)>tol:
                    self.box = (0,row,width,height)
                    if scale_location == 'Top':
                        crop_img = img.crop(self.box)
                    elif scale_location == 'Auto' and np.multiply(*self.box.size)>np.multiply(*crop_img.size):
                        crop_img = img.crop(self.box)
                    break
        elif scale_location == 'Right' or scale_location == 'Auto':
            for col in range(width):
                if np.mean(img_array[:,col]==0)>tol:
                    self.box = (0,0,col,height)
                    if scale_location == 'Right':
                        crop_img = img.crop(self.box)
                    elif scale_location == 'Auto' and np.multiply(*self.box.size)>np.multiply(*crop_img.size):
                        crop_img = img.crop(self.box)
                    break
        elif scale_location == 'Left' or scale_location == 'Auto':
            for col in reversed(range(width)):
                if np.mean(img_array[:,col]==0)>tol:
                    self.box = (col,0,width,height)
                    if scale_location == 'Left':
                        crop_img = img.crop(self.box)
                    elif scale_location == 'Auto' and np.multiply(*self.box.size)>np.multiply(*crop_img.size):
                        crop_img = img.crop(self.box)
                    break

        self.properties['scale_crop_box'] = self.box
        if crop_img:
            self.img_out = np.array(crop_img)
        else:
            self.img_out = self.mod_in.image()


class ColorMask(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(ColorMask,self).__init__(mod_in,img_item,properties)
        self.img_mask = None
        self.img_hist = self.img_item.getHistogram()

        self.wHistPlot = None
        self.lrItem = None

        self.wHistPlot = pg.PlotWidget()
        self.wHistPlot.plot(*self.img_hist)
        self.wHistPlot.setXRange(0,255)
        self.wHistPlot.hideAxis('left')

        self.lrItem = pg.LinearRegionItem((0,255),bounds=(0,255))
        self.lrItem.sigRegionChanged.connect(self.update_view)
        self.lrItem.sigRegionChangeFinished.connect(self.update_view)

        self.wHistPlot.addItem(self.lrItem)
        self.wHistPlot.setMouseEnabled(False,False)
        self.wHistPlot.setMaximumHeight(100)

    def to_dict(self):
        d = super(ColorMask,self).to_dict()
        d['img_mask'] = self.img_mask.tolist()
        d['LinearRegionItem'] = {'region':self.lrItem.getRegion()}
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(ColorMask,cls).from_dict(d,img_item)
        obj.img_mask = np.asarray(d['img_mask'])
        obj.lrItem.setRegion(d['LinearRegionItem']['region'])
        obj.update_image()
        obj.widget().hide()
        return obj

    def widget(self):
        return self.wHistPlot

    def update_image(self):
        minVal, maxVal = self.lrItem.getRegion()
        img = self.mod_in.image()
        self.img_mask = np.zeros_like(img)
        self.img_mask[np.logical_and(img>minVal,img<maxVal)] = 1
        self.img_out = img*self.img_mask+(1-self.img_mask)*255

    def name(self):
        return 'Color Mask'

class CannyEdgeDetection(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(CannyEdgeDetection,self).__init__(mod_in,img_item,properties)

        self.low_thresh = int(max(self.mod_in.image().flatten())*.1)
        self.high_thresh = int(max(self.mod_in.image().flatten())*.4)
        self.gauss_size = 5
        self.wToolBox = pg.LayoutWidget()
        self.wToolBox.layout.setAlignment(QtCore.Qt.AlignTop)

        self.wGaussEdit = QtGui.QLineEdit(str(self.gauss_size))
        self.wGaussEdit.setValidator(QtGui.QIntValidator(3,51))
        self.wGaussEdit.setFixedWidth(60)

        self.wLowSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wLowSlider.setMinimum(0)
        self.wLowSlider.setMaximum(255)
        self.wLowSlider.setSliderPosition(int(self.low_thresh))
        self.wLowEdit = QtGui.QLineEdit(str(self.low_thresh))
        self.wLowEdit.setFixedWidth(60)
        self.wLowEdit.setValidator(QtGui.QIntValidator(0,255))

        self.wHighSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wHighSlider.setMinimum(0)
        self.wHighSlider.setMaximum(255)
        self.wHighSlider.setSliderPosition(int(self.high_thresh))
        self.wHighEdit = QtGui.QLineEdit(str(self.high_thresh))
        self.wHighEdit.setFixedWidth(60)
        self.wHighEdit.setValidator(QtGui.QIntValidator(0,255))

        self.wGaussEdit.returnPressed.connect(self._update_sliders)
        self.wLowSlider.sliderReleased.connect(self._update_texts)
        self.wLowSlider.sliderMoved.connect(self._update_texts)
        self.wLowEdit.returnPressed.connect(self._update_sliders)
        self.wHighSlider.sliderReleased.connect(self._update_texts)
        self.wHighSlider.sliderMoved.connect(self._update_texts)
        self.wHighEdit.returnPressed.connect(self._update_sliders)

        self.wToolBox.addWidget(QtGui.QLabel('Gaussian Size'),0,0)
        self.wToolBox.addWidget(QtGui.QLabel('Low Threshold'),1,0)
        self.wToolBox.addWidget(QtGui.QLabel('High Threshold'),3,0)
        self.wToolBox.addWidget(self.wGaussEdit,0,1)
        self.wToolBox.addWidget(self.wLowEdit,1,1)
        self.wToolBox.addWidget(self.wHighEdit,3,1)
        self.wToolBox.addWidget(self.wLowSlider,2,0,1,2)
        self.wToolBox.addWidget(self.wHighSlider,4,0,1,2)

    def to_dict(self):
        d = super(CannyEdgeDetection,self).to_dict()
        d['canny_inputs'] = {
            'low_threshold': self.low_thresh,
            'high_threshold': self.high_thresh,
            'gaussian_size': self.gauss_size
            }
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(CannyEdgeDetection,cls).from_dict(d,img_item)
        obj.low_thresh = d['canny_inputs']['low_threshold']
        obj.high_thresh = d['canny_inputs']['high_threshold']
        obj.gauss_size = d['canny_inputs']['gaussian_size']

        obj.wLowEdit.setText(str(obj.low_thresh))
        obj.wHighEdit.setText(str(obj.high_thresh))
        obj.wGaussEdit.setText(str(obj.gauss_size))

        obj.wGaussEdit.setText(str(obj.gauss_size))
        obj.wLowSlider.setSliderPosition(obj.low_thresh)
        obj.wHighSlider.setSliderPosition(obj.high_thresh)

        obj.update_image()
        obj.widget().hide()
        return obj

    def name(self):
        return 'Canny Edge Detection'

    def _update_sliders(self):
        self.gauss_size = int(self.wGaussEdit.text())
        self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
        self.wGaussEdit.setText(str(self.gauss_size))
        self.low_thresh = int(self.wLowEdit.text())
        self.high_thresh = int(self.wHighEdit.text())

        self.wLowSlider.setSliderPosition(self.low_thresh)
        self.wHighSlider.setSliderPosition(self.high_thresh)

        self.update_view()

    def _update_texts(self):
        self.low_thresh = int(self.wLowSlider.value())
        self.high_thresh = int(self.wHighSlider.value())

        self.wLowEdit.setText(str(self.low_thresh))
        self.wHighEdit.setText(str(self.high_thresh))

        self.update_view()

    def update_image(self):
        self.img_out = cv2.GaussianBlur(self.mod_in.image(),(self.gauss_size,self.gauss_size),0)
        self.img_out = 255-cv2.Canny(self.img_out,self.low_thresh,self.high_thresh,L2gradient=True)

    def widget(self):
        return self.wToolBox

class Dilation(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(Dilation,self).__init__(mod_in,img_item,properties)
        self.size = 1
        self.wToolBox = pg.LayoutWidget()
        self.wToolBox.layout.setAlignment(QtCore.Qt.AlignTop)
        self.wSizeEdit = QtGui.QLineEdit(str(self.size))
        self.wSizeEdit.setValidator(QtGui.QIntValidator(1,20))
        self.wSizeEdit.setFixedWidth(60)
        self.wSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(20)
        self.wSizeSlider.setSliderPosition(int(self.size))

        # self.wSizeSlider.sliderReleased.connect(self._update_texts)
        self.wSizeSlider.valueChanged.connect(self._update_texts)
        self.wSizeEdit.returnPressed.connect(self._update_sliders)

        self.wToolBox.addWidget(QtGui.QLabel('Kernel Size'),0,0)
        self.wToolBox.addWidget(self.wSizeEdit,0,1)
        self.wToolBox.addWidget(self.wSizeSlider,1,0,1,2)

    def to_dict(self):
        d = super(Dilation,self).to_dict()
        d['size'] = self.size
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(Dilation,cls).from_dict(d,img_item)
        obj.size = d['size']
        obj.wSizeSlider.setSliderPosition(d['size'])
        obj._update_texts()
        obj.widget().hide()
        return obj

    def name(self):
        return 'Dilation'

    def _update_sliders(self):
        self.size = int(self.wSizeEdit.text())
        self.wSizeSlider.setSliderPosition(self.size)
        self.update_view()

    def _update_texts(self):
        self.size = int(self.wSizeSlider.value())
        self.wSizeEdit.setText(str(self.size))
        self.update_view()

    def update_image(self):
        self.img_out = cv2.erode(self.mod_in.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

    def widget(self):
        return self.wToolBox

class Erosion(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(Erosion,self).__init__(mod_in,img_item,properties)
        self.size = 1
        self.wToolBox = pg.LayoutWidget()
        self.wToolBox.layout.setAlignment(QtCore.Qt.AlignTop)
        self.wSizeEdit = QtGui.QLineEdit(str(self.size))
        self.wSizeEdit.setValidator(QtGui.QIntValidator(1,20))
        self.wSizeEdit.setFixedWidth(60)
        self.wSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(20)
        self.wSizeSlider.setSliderPosition(int(self.size))

        # self.wSizeSlider.sliderReleased.connect(self._update_texts)
        self.wSizeSlider.valueChanged.connect(self._update_texts)
        self.wSizeEdit.returnPressed.connect(self._update_sliders)

        self.wToolBox.addWidget(QtGui.QLabel('Kernel Size'),0,0)
        self.wToolBox.addWidget(self.wSizeEdit,0,1)
        self.wToolBox.addWidget(self.wSizeSlider,1,0,1,2)

    def to_dict(self):
        d = super(Erosion,self).to_dict()
        d['size'] = self.size
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(Erosion,cls).from_dict(d,img_item)
        obj.size = d['size']
        obj.wSizeSlider.setSliderPosition(d['size'])
        obj._update_texts()
        obj.widget().hide()
        return obj

    def name(self):
        return 'Erosion'

    def _update_sliders(self):
        self.size = int(self.wSizeEdit.text())
        self.wSizeSlider.setSliderPosition(self.size)
        self.update_view()

    def _update_texts(self):
        self.size = int(self.wSizeSlider.value())
        self.wSizeEdit.setText(str(self.size))
        self.update_view()

    def update_image(self):
        self.img_out = cv2.dilate(self.mod_in.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

    def widget(self):
        return self.wToolBox

class BinaryMask(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(BinaryMask,self).__init__(mod_in,img_item,properties)

    def to_dict(self):
        d = super(BinaryMask,self).to_dict()
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(BinaryMask,cls).from_dict(d,img_item)
        obj.update_image()
        return obj

    def name(self):
        return 'Binary Mask'

    def update_image(self):
        self.img_out = self.mod_in.image()
        self.img_out[self.img_out < 255] = 0

class FindContours(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(FindContours,self).__init__(mod_in,img_item,properties)
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)
        self.img_inv = self.mod_in.image()
        self.img_inv[self.img_inv < 255] = 0
        self.img_inv = 255 - self.img_inv

        self.tol = 0.04
        self.wTolEdit = QtGui.QLineEdit(str(self.tol))
        self.wTolEdit.setValidator(QtGui.QDoubleValidator(0,1,3))
        self.wTolEdit.setFixedWidth(60)

        self.lowVert = 6
        self.wLowEdit = QtGui.QLineEdit(str(self.lowVert))
        self.wLowEdit.setValidator(QtGui.QIntValidator(3,100))
        self.wLowEdit.setFixedWidth(60)

        self.highVert = 6
        self.wHighEdit = QtGui.QLineEdit(str(self.highVert))
        self.wHighEdit.setValidator(QtGui.QIntValidator(3,100))
        self.wHighEdit.setFixedWidth(60)

        self.areaThresh = 0.5
        self.wThreshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wThreshSlider.setMinimum(0)
        self.wThreshSlider.setMaximum(100)
        self.wThreshSlider.setSliderPosition(50)

        major = cv2.__version__.split('.')[0]
        if major == 3:
            self._img, self.contours, self.hierarchy = cv2.findContours(
                self.img_inv,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)
        else:
            self.contours, self.hierarchy = cv2.findContours(
                self.img_inv,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)

        self.contour_dict = {}
        self.contour_area_max = 1

        self.wContourList = QtGui.QListWidget()
        self.wContourList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        for i,cnt in enumerate(self.contours):
            key = '%d Contour'%i
            self.contour_dict[key] = {}
            self.contour_dict[key]['index'] = i
            self.contour_dict[key]['list_item'] = QtGui.QListWidgetItem(key)
            self.contour_dict[key]['contour'] = cnt
            self.contour_dict[key]['area'] = cv2.contourArea(cnt,oriented=True)
            if abs(self.contour_dict[key]['area']) > self.contour_area_max:
                self.contour_area_max = abs(self.contour_dict[key]['area'])
        self.update_tol()

        self.wContourList.itemSelectionChanged.connect(self.update_view)
        self.wContourList.itemClicked.connect(self.update_view)
        self.wTolEdit.returnPressed.connect(self.update_tol)
        self.wLowEdit.returnPressed.connect(self.update_tol)
        self.wHighEdit.returnPressed.connect(self.update_tol)
        self.wThreshSlider.valueChanged.connect(self.update_tol)

        if len(self.contour_dict.keys())>0:
            self.wContourList.setCurrentItem(self.contour_dict['0 Contour']['list_item'])

        self.wLayout.addWidget(self.wContourList,0,0,2,1)
        self.wLayout.addWidget(QtGui.QLabel('Polygon Tolerance:'),3,0)
        self.wLayout.addWidget(self.wTolEdit,3,1)
        self.wLayout.addWidget(QtGui.QLabel('Vertex Tolerance:'),4,0)
        self.wLayout.addWidget(self.wLowEdit,4,1)
        self.wLayout.addWidget(self.wHighEdit,4,2)
        self.wLayout.addWidget(QtGui.QLabel('Contour Area Tolerance:'),5,0)
        self.wLayout.addWidget(self.wThreshSlider,6,0,1,3)

        self.update_view()

    def to_dict(self):
        d = super(FindContours,self).to_dict()
        d['line_tolerance'] = self.tol
        d['low_vertex_threshold'] = self.lowVert
        d['high_vertex_threshold'] = self.highVert
        d['contour_area_threshold'] = self.areaThresh
        d['threshold_slider_tick'] = self.wThreshSlider.value()
        d['contours'] = self.contours
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(FindContours,cls).from_dict(d,img_item)
        obj.tol = d['line_tolerance']
        obj.lowVert = d['low_vertex_threshold']
        obj.highVert = d['high_vertex_threshold']
        obj.areaThresh = d['contour_area_threshold']

        obj.wTolEdit.setText(str(obj.tol))
        obj.wLowEdit.setText(str(obj.lowVert))
        obj.wHighEdit.setText(str(obj.highVert))
        obj.wThreshSlider.setSliderPosition(d['threshold_slider_tick'])
        obj.update_image()

        obj.widget().hide()
        return obj

    def update_image(self):
        self.img_out = self.mod_in.image()
        selection = self.wContourList.selectedItems()
        if len(selection) == 1:
            cnt_key = selection[0].text()
            accept, approx = self.detect_poly(self.contour_dict[cnt_key]['contour'])
            cv2.drawContours(self.img_out,[approx],0,thickness=2,color=(0,255,0))

    def detect_poly(self,cnt):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, self.tol * peri, True)
        return len(approx) >= self.lowVert and len(approx) <= self.highVert, approx

    def update_tol(self):
        self.tol = float(self.wTolEdit.text())
        self.lowVert = float(self.wLowEdit.text())
        self.highVert = float(self.wHighEdit.text())
        self.areaThresh = float(self.wThreshSlider.value())/100.
        self.wContourList.clear()
        for key in self.contour_dict.keys():
            cnt = self.contour_dict[key]['contour']
            area = self.contour_dict[key]['area']
            accept, approx = self.detect_poly(cnt)
            if accept and area < 0 and abs(area/self.contour_area_max) >= self.areaThresh:
                self.contour_dict[key]['list_item'] = QtGui.QListWidgetItem(key)
                self.wContourList.addItem(self.contour_dict[key]['list_item'])

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Find Contours'

class Blur(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(Blur,self).__init__(mod_in,img_item,properties)
        self.gauss_size = 5
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)
        self.wGaussEdit = QtGui.QLineEdit(str(self.gauss_size))
        self.wGaussEdit.setFixedWidth(100)
        self.wGaussEdit.setValidator(QtGui.QIntValidator(3,51))

        self.wLayout.addWidget(QtGui.QLabel('Gaussian Size:'),0,0)
        self.wLayout.addWidget(self.wGaussEdit,0,1)

        self.update_view()
        self.wGaussEdit.returnPressed.connect(self.update_view)

    def to_dict(self):
        d = super(Blur,self).to_dict()
        d['gaussian_size'] = self.gauss_size
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(Blur,cls).from_dict(d,img_item)
        obj.gauss_size = d['gaussian_size']
        obj.wGaussEdit.setText(str(obj.gauss_size))
        obj.update_image()
        obj.widget().hide()

        return obj

    def update_image(self):
        self.gauss_size = int(self.wGaussEdit.text())
        self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
        self.wGaussEdit.setText(str(self.gauss_size))
        self.img_out = cv2.GaussianBlur(self.mod_in.image(),(self.gauss_size,self.gauss_size),0)

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Blur'

class TemplateMatchingWidget(Modification):
    imageChanged = QtCore.pyqtSignal(object)
    def __init__(self,mod_in,img_item,vbox,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,mod_in,img_item,properties,parent=parent)
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.mod_in.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.mod_in.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in3d.copy()

        self.threshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(1000)
        self.threshSlider.setSliderPosition(100)
        self.threshSlider.valueChanged.connect(self.update_view)

        self.sizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sizeSlider.setMinimum(2)
        self.sizeSlider.setMaximum(30)
        self.sizeSlider.setSliderPosition(15)
        self.sizeSlider.valueChanged.connect(lambda v: self.roi.setSize([2*v,2*v]))

        self.roi = pg.ROI(
            pos=(0,0),
            size=(20,20),
            removable=True,
            pen=pg.mkPen(color='r',width=3),
            maxBounds=self.img_item.boundingRect(),
            scaleSnap=True,
            snapSize=2)
        self.roi.sigRegionChangeFinished.connect(self.update_view)
        vbox.addItem(self.roi)

        main_widget = QtWidgets.QWidget()
        layout = QtGui.QGridLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(QtWidgets.QLabel("Threshold:"),0,0)
        layout.addWidget(self.threshSlider,0,1)
        layout.addWidget(QtWidgets.QLabel("Template Size:"),1,0)
        layout.addWidget(self.sizeSlider,1,1)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)


    def update_image(self,threshold=100):
        region = self.roi.getArrayRegion(self.img_in,self.img_item).astype(np.uint8)
        x,y = region.shape
        padded_image = cv2.copyMakeBorder(self.img_in,int(y/2-1),int(y/2),int(x/2-1),int(x/2),cv2.BORDER_REFLECT_101)
        res = cv2.matchTemplate(padded_image,region,cv2.TM_SQDIFF_NORMED)

        threshold = np.logspace(-3,0,1000)[threshold-1]
        
        self._mask[...] = False
        self._mask[res < threshold] = True
        if isinstance(self._mask_in,np.ndarray):
            self._mask = np.logical_or(self._mask,self._mask_in)

        self.roi_img = mask_color_img(self.img_in3d, self._mask, color=[0, 0, 255], alpha=0.3)

        self.img_out = self.img_in.copy()
        self.img_out[np.logical_not(self._mask)] = 255

        self.imageChanged.emit(self.img_out)

    def update_view(self):
        self.update_image(self.threshSlider.value())
        self.img_item.setImage(self.roi_img,levels=(0,255))

    def focus(self):
        self.roi.show()
        self.imageChanged.emit(self.roi_img)

    def unfocus(self):
        self.roi.hide()

    @property
    def mask(self):
        return self._mask


class CustomFilter(Modification):
    imageChanged = QtCore.pyqtSignal(object)
    def __init__(self,mod_in,img_item,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,mod_in,img_item,properties,parent=parent)
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.mod_in.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.mod_in.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in3d.copy()

        self.sizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sizeSlider.setMinimum(2)
        self.sizeSlider.setMaximum(100)
        self.sizeSlider.setSliderPosition(20)
        self.sizeSlider.valueChanged.connect(self.updateKernelCursor)

        main_widget = QtWidgets.QWidget()
        layout = QtGui.QGridLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(QtWidgets.QLabel("Cursor Size:"),0,0)
        layout.addWidget(self.sizeSlider,0,1)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)

    def updateKernelCursor(self,radius,kern_val=1):
        kern = np.full((2*radius,2*radius),1-kern_val,dtype=bool)
        ctr = (radius,radius)

        rr, cc = skcircle(radius,radius,radius)
        kern[rr,cc] = kern_val

        self.img_item.setDrawKernel(kern,center=ctr)
        self.img_item.updateCursor(radius)

    def update_view(self,slices=None,mask=None,comparator=np.logical_or):
        if slices is not None and mask is not None:
            self._mask[slices] = comparator(mask,self._mask[slices])

            if isinstance(self._mask_in,np.ndarray):
                self._mask = comparator(self._mask_in,self._mask)

            self.roi_img = mask_color_img(
                self.img_in3d, 
                self._mask, 
                color=[0, 0, 255], 
                alpha=0.3)

            self.img_out = self.img_in.copy()
            self.img_out[np.logical_not(self._mask)] = 255
            self.imageChanged.emit(self.img_out)
            self.img_item.setImage(self.roi_img,levels=(0,255))
        

    def focus(self):
        self.updateKernelCursor(self.sizeSlider.value())
        self.img_item.imageUpdateSignal.connect(self.update_view)


    def unfocus(self):
        self.img_item.setDrawKernel()
        self.img_item.resetCursor()
        self.img_item.disconnect()

    @property
    def mask(self):
        return self._mask

class EraseFilter(CustomFilter):
    def __init__(self,*args,**kwargs):
        CustomFilter.__init__(self,*args,**kwargs)
        self._mask = np.ones_like(self.mod_in.image(),dtype=bool)

    def update_view(self,slices=None,mask=None):
        CustomFilter.update_view(self,slices,mask,comparator=np.logical_and)

    def updateKernelCursor(self,radius):
        CustomFilter.updateKernelCursor(self,radius,kern_val=0)

class ClusterFilter(Modification):
    imageChanged = QtCore.pyqtSignal(object)
    def __init__(self,mod_in,img_item,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,mod_in,img_item,properties,parent=parent)
        self._clusters = None
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.mod_in.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.mod_in.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in.copy()

        self.cluster_list = QtWidgets.QListWidget()
        self.cluster_list.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        self.wsize_edit = QtWidgets.QLineEdit()
        self.wsize_edit.setValidator(QtGui.QIntValidator(1,50))
        self.wsize_edit.setText("15")
        self.run_btn = QtWidgets.QPushButton("Run")

    def focus(self):
        pass

    def unfocus(self):
        pass

    @property
    def mask(self):
        return self._mask


    def pad(self,img,wsize,stride=1):
        height,width = img.shape
        if stride == 'block':
            adj = 0
            stride = wsize
        else:
            adj = 1

        px = wsize - height % stride - adj
        if px % 2 == 0:
            px = int(px/2)
            px = (px,px)
        else:
            px = int((px-1)/2)
            px = (px,px+1)

        py = wsize - width % stride - adj
        if py % 2 == 0:
            py = int(py/2)
            py = (py,py)
        else:
            py = int((py-1)/2)
            py = (py,py+1)

        return util.pad(img,pad_width=(px,py),mode='symmetric')

    def update_image(self):
        selected_items = self.cluster_list.selectedItems()
        self._mask = np.zeros_like(self.img_in,dtype=bool)
        for item in selected_items:
            self._mask[self._clusters==item.data(QtCore.Qt.UserRole)] = True
        if self._mask_in is not None:
            self._mask = np.logical_or(self._mask,self._mask_in)

        self.roi_img = mask_color_img(
            self.img_in3d, 
            self._mask, 
            color=[0, 0, 255], 
            alpha=0.3)

        self.img_out = self.img_in.copy()
        self.img_out[np.logical_not(self._mask)] = 255

        self.update_view()

    def update_view(self):
        self.imageChanged.emit(self.img_out)
        self.img_item.setImage(self.roi_img,levels=(0,255))

    def update_list(self):
        self.cluster_list.clear()
        if self._clusters is not None:
            labels, counts = np.unique(self._clusters,return_counts=True)
            fractions = np.round(counts/counts.sum(),3)
            order = np.argsort(fractions)[::-1]

            for label, fraction in zip(labels[order],fractions[order]):
                item = QtGui.QListWidgetItem(str(fraction))
                item.setData(QtCore.Qt.UserRole,label)
                self.cluster_list.addItem(item)

    def filter(self):
        pass

class KMeansFilter(ClusterFilter):
    def __init__(self,*args,**kwargs):
        ClusterFilter.__init__(self,*args,**kwargs)
        self.n_clusters_edit = QtWidgets.QLineEdit()
        self.n_clusters_edit.setValidator(QtGui.QIntValidator(2,20))
        self.n_clusters_edit.setText("2")

        self.stride_edit = QtWidgets.QLineEdit()
        self.stride_edit.setValidator(QtGui.QIntValidator(1,30))
        self.stride_edit.setText("3")

        main_widget = QtWidgets.QWidget()
        layout = QtGui.QGridLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(QtWidgets.QLabel("Window Size:"),0,0)
        layout.addWidget(self.wsize_edit,0,1)
        layout.addWidget(QtWidgets.QLabel("# Clusters:"),1,0)
        layout.addWidget(self.n_clusters_edit,1,1)
        layout.addWidget(QtWidgets.QLabel("Stride:"),2,0)
        layout.addWidget(self.stride_edit,2,1)
        layout.addWidget(self.run_btn,3,0)
        layout.addWidget(self.cluster_list,4,0,1,2)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)

        self.run_btn.clicked.connect(self.filter)
        self.cluster_list.itemSelectionChanged.connect(self.update_image)

    def filter(self):
        wsize = int('0'+self.wsize_edit.text())
        if wsize % 2 == 0:
            wsize -= 1
        self.wsize_edit.setText(str(wsize))

        stride = int("0"+self.stride_edit.text())

        n_clusters = int('0'+self.n_clusters_edit.text())

        if n_clusters >= 2 and wsize >= 1 and stride >= 1:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters)
            X=util.view_as_windows(
                self.pad(self.img_in,wsize=wsize,stride=stride),
                window_shape=(wsize,wsize),
                step=stride)
            mask_dim = X.shape[:2]
            X=X.reshape(-1,wsize**2)

            kmeans = kmeans.fit(X)
            mask = kmeans.labels_.reshape(*mask_dim)
            mask = Image.fromarray(mask)
            self._clusters = np.array(mask.resize(self.img_in.shape[::-1])).astype(np.uint8)
            del X

            self.update_list()
            self.update_view()


class GMMFilter(ClusterFilter):
    def __init__(self,*args,**kwargs):
        ClusterFilter.__init__(self,*args,**kwargs)
        self.n_components_edit = QtWidgets.QLineEdit()
        self.n_components_edit.setValidator(QtGui.QIntValidator(2,20))
        self.n_components_edit.setText("2")

        main_widget = QtWidgets.QWidget()
        layout = QtGui.QGridLayout(self)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(QtWidgets.QLabel("Window Size:"),0,0)
        layout.addWidget(self.wsize_edit,0,1)
        layout.addWidget(QtWidgets.QLabel("# Components:"),1,0)
        layout.addWidget(self.n_components_edit,1,1)
        layout.addWidget(self.run_btn,2,0)
        layout.addWidget(self.cluster_list,3,0,1,2)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)

        self.run_btn.clicked.connect(self.filter)
        self.cluster_list.itemSelectionChanged.connect(self.update_image)

    def filter(self):
        wsize = int('0'+self.wsize_edit.text())
        if wsize % 2 == 0:
            wsize -= 1
        self.wsize_edit.setText(str(wsize))

        n_components = int('0'+self.n_components_edit.text())

        if n_components >= 2 and wsize >= 1:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                n_init=10
                )
            X = util.view_as_blocks(
                self.pad(self.img_in,wsize=wsize,stride='block'),
                block_shape=(wsize,wsize)).reshape(-1,wsize**2)
            gmm.fit(X)
            del X
            mod_img=util.view_as_windows(self.pad(self.img_in,wsize=wsize),window_shape=(wsize,wsize))

            clusters = np.zeros(mod_img.shape[0]*mod_img.shape[1])
            for i in range(mod_img.shape[0]):
                x = mod_img[i,...].reshape(-1,wsize**2)
                
                d = x.shape[0]
                clusters[i*d:(i+1)*d] = gmm.predict(x)

            self._clusters = clusters.reshape(*self.img_in.shape).astype(np.uint8)

            self.update_list()
            self.update_view()


class FilterPattern(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(FilterPattern,self).__init__(mod_in,img_item,properties)

        self._maskClasses = OrderedDict()
        self._maskClasses['Template Match'] = TemplateMatchingWidget
        self._maskClasses['Gaussian Mixture'] = GMMFilter
        self._maskClasses['K-Means'] = KMeansFilter
        self._maskClasses['Custom'] = CustomFilter
        self._maskClasses['Erase'] = EraseFilter

        self.wFilterType = QtWidgets.QComboBox()
        self.wFilterType.addItems(list(self._maskClasses.keys()))

        self.wFilterList = QtGui.QListWidget()
        self.wFilterList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.wAdd = QtGui.QPushButton('Add Filter')
        self.wRemove = QtGui.QPushButton('Remove Layer')
        self.wExportMask = QtGui.QPushButton('Export Mask')

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = SmartImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        self.wImgBox_VB.sigResized.connect(lambda v: self.wImgROI.updateCursor())
        self.wImgBox_VB.sigTransformChanged.connect(lambda v: self.wImgROI.updateCursor())
        
        self.blankWidget = QtWidgets.QWidget()
        self.stackedControl = QtWidgets.QStackedWidget()

        self.toggleControl = QtWidgets.QStackedWidget()
        self.toggleControl.addWidget(self.blankWidget)
        self.toggleControl.addWidget(self.stackedControl)

        layer_layout = QtGui.QGridLayout()
        layer_layout.setAlignment(QtCore.Qt.AlignTop)
        layer_layout.addWidget(self.wFilterList,0,0,6,1)
        layer_layout.addWidget(self.wFilterType,0,1)
        layer_layout.addWidget(self.wAdd,1,1)
        layer_layout.addWidget(self.wRemove,2,1)
        layer_layout.addWidget(self.wExportMask,4,1)
        
        main_widget = QtWidgets.QWidget()
        main_layout = QtGui.QGridLayout(main_widget)
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        main_layout.addLayout(layer_layout,0,0)
        main_layout.addWidget(self.toggleControl,1,0)
        main_layout.addWidget(self.wImgBox,2,0)

        self.setWidget(main_widget)

        self.wFilterList.currentRowChanged.connect(self.update_view)
        self.wFilterList.itemActivated.connect(self.update_view)
        self.wAdd.clicked.connect(self.add)
        self.wRemove.clicked.connect(self.delete)
        self.wExportMask.clicked.connect(self.export)

    def name(self):
        return "Filter Pattern"

    def update_properties(self):
        count = self.stackedControl.count()
        if count > 0:
            self.properties["mask_total"] = self.stackedControl.widget(count-1).mask.tolist()

    def image(self):
        if self.wFilterList.count()>0:
            return self.stackedControl.widget(self.stackedControl.count()-1).image()
        else:
            return self.img_out.copy()

    def update_view(self,value=None):   
        if isinstance(value,QtGui.QListWidgetItem):
            row = self.wFilterList.row(value)
        elif isinstance(value,int):
            row = value
        else:
            row = max(self.wFilterList.count()-1,-1)


        if row>-1:
            widget = self.stackedControl.widget(row)
            widget.update_view()
            for i in range(self.stackedControl.count()):
                w = self.stackedControl.widget(i)
                if i == row:
                    w.focus()
                else:
                    w.unfocus()
            if row == self.wFilterList.count()-1:
                self.toggleControl.setCurrentWidget(self.stackedControl)
                self.stackedControl.setCurrentIndex(row)
            else:
                self.toggleControl.setCurrentWidget(self.blankWidget)

            self.img_item.setImage(widget.image(),levels=(0,255))
        else:
            self.toggleControl.setCurrentWidget(self.blankWidget)
            self.img_item.setImage(self.image(),levels=(0,255))
            self.wImgROI.setImage(self.image(),levels=(0,255))

    def add(self):
        method = self.wFilterType.currentText()
        maskClass = self._maskClasses[method]

        if self.wFilterList.count()>0:
            in_mod = self.stackedControl.widget(self.stackedControl.count()-1)
            mask_in = in_mod._mask
        else:
            in_mod = self.mod_in
            mask_in = None

        if method == 'Template Match':
            mask_widget = maskClass(mod_in=in_mod,img_item=self.wImgROI,vbox=self.wImgBox_VB,mask_in=mask_in,img_in=self.mod_in.image())
        else:
            mask_widget = maskClass(mod_in=in_mod,img_item=self.wImgROI,mask_in=mask_in,img_in=self.mod_in.image())

        mask_widget.imageChanged.connect(lambda img: self.img_item.setImage(img,levels=(0,255)))
        mask_widget.imageChanged.connect(lambda img: self.update_properties())
        self.stackedControl.addWidget(mask_widget)
        self.wFilterList.addItem(method)

        self.update_view()

    def delete(self):
        if self.wFilterList.count()>0:
            index = self.wFilterList.count()-1
            w = self.stackedControl.widget(index)
            w.unfocus()
            if isinstance(w,TemplateMatchingWidget):
                self.wImgBox_VB.removeItem(w.roi)
            self.wFilterList.takeItem(index)
            self.stackedControl.removeWidget(w)
            self.update_view()

    def export(self):
        export_mask = np.zeros_like(self.image(),dtype=np.uint8)
        if self.stackedControl.count()>0:
            export_mask = self.stackedControl.widget(self.stackedControl.count()-1).mask.astype(np.uint8)*255

        default_name = "untitled"
        if self.properties['mode'] == 'local':
            path = os.path.join(os.getcwd(),default_name+"_mask.png")
            name = QtWidgets.QFileDialog.getSaveFileName(None, 
                "Export Image", 
                path, 
                "PNG File (*.png)",
                "PNG File (*.png)")[0]
            if name != '' and check_extension(name, [".png"]):
                cv2.imwrite(name,export_mask)
        elif self.properties["mode"] == 'nanohub':
            name = default_name+"_mask.png"
            cv2.imwrite(name,export_mask)
            subprocess.check_output('exportfile %s'%name,shell=True)
            try:
                os.remove(name)
            except:
                pass
        else:
            return

class Crop(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(Crop,self).__init__(mod_in,img_item,properties)
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        # self.wImgBox_VB.setMouseEnabled(False,False)

        self.roi = pg.ROI(
            pos=(0,0),
            size=(20,20),
            removable=True,
            pen=pg.mkPen(color='r',width=2),
            maxBounds=self.wImgROI.boundingRect(),)
        self.roi.addScaleHandle(pos=(1,1),center=(0,0))
        self.wImgBox_VB.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_view)

        self.wLayout.addWidget(self.wImgBox,0,0)

    def to_dict(self):
        d = super(Crop,self).to_dict()
        d['roi_state'] = self.roi.saveState()
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(Crop,cls).from_dict(d,img_item)
        obj.roi.setState(d['roi_state'])
        obj.update_image()
        obj.widget().hide()

        return obj

    def update_image(self):
        self.img_out,coords = self.roi.getArrayRegion(self.wImgROI.image,self.wImgROI,returnMappedCoords=True)
        self.img_out = self.img_out.astype(np.uint8)

        self.properties['crop_coords'] = coords.tolist()

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Crop'

class HoughTransform(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(HoughTransform,self).__init__(mod_in,img_item,properties)
        self.inv_img = 255 - self.mod_in.image()
        self.img_out = self.mod_in.image()
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)
        self.thresh_tick = 100
        self.min_angle = 10
        self.min_dist = 9
        self.line_length = 50
        self.line_gap = 10
        self.hspace,self.angles,self.distances = transform.hough_line(self.inv_img)
        self.bgr_img = cv2.cvtColor(self.img_out,cv2.COLOR_GRAY2BGR)
        self.bgr_hough = 255-np.round(self.hspace/np.max(self.hspace)*255).astype(np.uint8)
        self.bgr_hough = cv2.cvtColor(self.bgr_hough,cv2.COLOR_GRAY2BGR)


        self.properties['hough_transform'] = {
            'angles': self.angles,
            'distances': self.distances,
            'hspace': self.hspace
        }

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wHough = pg.ImageItem()
        # self.wHough.setImage(self.hspace,levels=(0,255))
        self.wImgBox_VB.addItem(self.wHough)
        # self.wImgBox_VB.setAspectLocked(True)
        # self.wImgBox_VB.setMouseEnabled(False,False)

        self.wHistPlot = pg.PlotWidget(title='Angle Histogram')
        self.wHistPlot.setXRange(0,180)
        self.wHistPlot.hideAxis('left')

        self.wMinAngleSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wMinAngleSlider.setMinimum(5)
        self.wMinAngleSlider.setMaximum(180)
        self.wMinAngleSlider.setSliderPosition(self.min_angle)

        self.wMinDistSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wMinDistSlider.setMinimum(5)
        self.wMinDistSlider.setMaximum(200)
        self.wMinDistSlider.setSliderPosition(self.min_dist)

        self.wThreshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wThreshSlider.setMinimum(0)
        self.wThreshSlider.setMaximum(200)
        self.wThreshSlider.setSliderPosition(self.thresh_tick)

        self.wLengthSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wLengthSlider.setMinimum(10)
        self.wLengthSlider.setMaximum(200)
        self.wLengthSlider.setSliderPosition(self.line_length)

        self.wGapSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wGapSlider.setMinimum(5)
        self.wGapSlider.setMaximum(100)
        self.wGapSlider.setSliderPosition(self.line_gap)

        self.wLayout.addWidget(QtGui.QLabel('Minimum Angle:'),0,0)
        self.wLayout.addWidget(self.wMinAngleSlider,0,1)
        self.wLayout.addWidget(QtGui.QLabel('Minimum Distance:'),1,0)
        self.wLayout.addWidget(self.wMinDistSlider,1,1)
        self.wLayout.addWidget(QtGui.QLabel('Threshold:'),2,0)
        self.wLayout.addWidget(self.wThreshSlider,2,1)
        self.wLayout.addWidget(self.wImgBox,3,0,2,2)
        self.wLayout.addWidget(QtGui.QLabel('Minimum Line Length:'),5,0)
        self.wLayout.addWidget(self.wLengthSlider,5,1)
        self.wLayout.addWidget(QtGui.QLabel('Minimum Gap Length:'),6,0)
        self.wLayout.addWidget(self.wGapSlider,6,1)
        self.wLayout.addWidget(self.wHistPlot,7,0,2,2)

        self.wThreshSlider.valueChanged.connect(self.update_image)
        self.wMinAngleSlider.valueChanged.connect(self.update_image)
        self.wMinDistSlider.valueChanged.connect(self.update_image)
        self.wLengthSlider.valueChanged.connect(self.update_image)
        self.wGapSlider.valueChanged.connect(self.update_image)

        self.update_view()

    def to_dict(self):
        d = super(HoughTransform,self).to_dict()
        d['hough_line_peaks'] = {
            'min_distance': self.min_dist,
            'min_angle': self.min_angle,
            'threshold': self.threshold
        }
        d['probabilistic_hough_line'] = {
            'threshold': self.threshold,
            'line_length': self.line_length,
            'line_gap': self.line_gap
        }
        d['threshold_slider_tick'] = self.thresh_tick
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(HoughTransform,cls).from_dict(d,img_item)
        obj.wMinAngleSlider.setSliderPosition(str(d['hough_line_peaks']['min_angle']))
        obj.wMinDistSlider.setSliderPosition(str(d['hough_line_peaks']['min_distance']))
        obj.wThreshSlider.setSliderPosition(str(d['hough_line_peaks']['threshold']))
        obj.wLengthSlider.setSliderPosition(str(d['probabilistic_hough_line']['line_length']))
        obj.wGapSlider.setSliderPosition(str(d['probabilistic_hough_line']['line_gap']))
        obj.update_image()
        obj.widget().hide()

        return obj

    def update_image(self):
        self.thresh_tick = int(self.wThreshSlider.value())
        self.threshold = int(np.max(self.hspace)*self.wThreshSlider.value()/200)
        self.min_angle = int(self.wMinAngleSlider.value())
        self.min_dist = int(self.wMinDistSlider.value())
        self.line_length = int(self.wLengthSlider.value())
        self.line_gap = int(self.wGapSlider.value())

        accum, angles, dists = transform.hough_line_peaks(
            self.hspace,
            self.angles,
            self.distances,
            min_distance = self.min_dist,
            min_angle = self.min_angle,
            threshold = self.threshold)

        # angle_diffs = []
        # for i,a1 in enumerate(angles):
        #   for j,a2 in enumerate(angles):
        #     if i < j:
        #       angle_diffs.append(abs(a1-a2)*180)

        y,x = np.histogram(np.array(angles)*180,bins=np.linspace(0,180,180))
        self.wHistPlot.clear()
        self.wHistPlot.plot(x,y,stepMode=True,fillLevel=0,brush=(0,0,255,150))

        lines = transform.probabilistic_hough_line(
            self.inv_img,
            threshold=self.threshold,
            line_length=self.line_length,
            line_gap=self.line_gap)

        self.bgr_hough = 255-np.round(self.hspace/np.max(self.hspace)*255).astype(np.uint8)
        self.bgr_hough = cv2.cvtColor(self.bgr_hough,cv2.COLOR_GRAY2BGR)

        for a,d in zip(angles,dists):
            angle_idx = np.nonzero(a == self.angles)[0]
            dist_idx = np.nonzero(d == self.distances)[0]
            cv2.circle(self.bgr_hough,center=(angle_idx,dist_idx),radius=5,color=(0,0,255),thickness=-1)

        self.bgr_img = self.mod_in.image()
        self.bgr_img = cv2.cvtColor(self.bgr_img,cv2.COLOR_GRAY2BGR)

        for p1,p2 in lines:
            cv2.line(self.bgr_img,p1,p2,color=(0,0,255),thickness=2)


        self.update_view()

    def update_view(self):
        self.wHough.setImage(self.bgr_hough)
        self.img_item.setImage(self.bgr_img)


        return self.properties

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Hough Transform'

class DomainCenters(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(DomainCenters,self).__init__(mod_in,img_item,properties)
        self.domain_centers = OrderedDict()
        self.mask = None
        self.radius = 10
        for mod in self.tolist():
            if isinstance(mod,FilterPattern):
                prop = mod.properties
                if 'mask_total' in prop.keys():
                    self.mask = np.array(prop['mask_total'])
        if self.mask is None:
            raise RuntimeError("Cannot find a mask! Use 'Filter Pattern' to generate one and try again.")
        self.img_out = self.mod_in.image()

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = SmartImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)

        self.wImgROI.setImage(self.img_out,levels=(0,255))
        self.wImgROI.setClickOnly(True)
        self.updateKernel(self.radius)

        self.domain_list = QtWidgets.QListWidget()
        self.deleteBtn = QtWidgets.QPushButton("Delete")
        self.exportBtn = QtWidgets.QPushButton("Export")

        main_widget = QtWidgets.QWidget()

        layer_layout = QtGui.QGridLayout(main_widget)
        layer_layout.setAlignment(QtCore.Qt.AlignTop)
        layer_layout.addWidget(self.domain_list,0,0)
        layer_layout.addWidget(self.deleteBtn,1,0)
        layer_layout.addWidget(self.wImgBox,2,0)
        layer_layout.addWidget(self.exportBtn,3,0)

        self.setWidget(main_widget)

        self.wImgROI.imageUpdateSignal.connect(self.update_image)
        self.domain_list.currentRowChanged.connect(self.update_view)
        self.deleteBtn.clicked.connect(lambda: self.deleteDomain(self.domain_list.currentRow()))
        self.exportBtn.clicked.connect(self.export)

    def updateKernel(self,radius,kern_val=1):
        kern = np.full((2*radius,2*radius),1-kern_val,dtype=bool)
        ctr = (radius,radius)

        rr, cc = skcircle(radius,radius,radius)
        kern[rr,cc] = kern_val

        self.wImgROI.setDrawKernel(kern,center=ctr)

    def update_properties(self):
        self.properties["domain_centers"] = list(self.domain_centers.keys())

    def export(self):
        default_name = "untitled.json"
        if self.properties['mode'] == 'local':
            name = QtWidgets.QFileDialog.getSaveFileName(None, 
                "Export", 
                path, 
                "JSON File (*.json)",
                "JSON File (*.json)")[0]
            if name != '' and check_extension(name, [".json"]):
                with open(name,'w') as f:
                    json.dump(list(self.domain_centers.keys()),f)
        elif self.properties["mode"] == 'nanohub':
            with open(default_name,'w') as f:
                json.dump(list(self.domain_centers.keys()),f)
            subprocess.check_output('exportfile %s'%(default_name),shell=True)
            try:
                os.remove(default_name)
            except:
                pass
        else:
            return

    def deleteDomain(self,index=None):
        if index is not None and index>=0:
            key = list(self.domain_centers.keys())[index]
            del self.domain_centers[key]

            self.update_view()
            self.update_properties()

    def update_image(self,slices,mask):
        row_slice, col_slice = slices
        r = int((row_slice.start + row_slice.stop)/2)
        c = int((col_slice.start + col_slice.stop)/2)
        self.domain_centers[(r,c)] = {'slices':slices,'mask':mask}

        self.update_view()
        self.update_properties()

    def update_view(self, index=None):
        domain_mask = np.zeros_like(self.img_out,dtype=bool)
        select_mask = np.zeros_like(self.img_out,dtype=bool)
        if index is None:
            self.domain_list.clear()
        for c, center in enumerate(self.domain_centers.keys()):
            if index is None:
                x,y = center
                self.domain_list.addItem("(%s,%s)"%(x,y))

            slices = self.domain_centers[center]["slices"]
            mask = self.domain_centers[center]["mask"]
            if index is not None and index == c:
                select_mask[slices] = mask
            else:
                domain_mask[slices] = mask

        img3d = np.dstack((self.img_out,self.img_out,self.img_out))
        if domain_mask.any():
            img3d = mask_color_img(
                img3d, 
                domain_mask, 
                color=[0, 0, 255], 
                alpha=0.3)
        if select_mask.any():
            img3d = mask_color_img(
                img3d, 
                select_mask, 
                color=[255, 0, 0], 
                alpha=0.3)
        self.wImgROI.setImage(img3d,levels=(0,255))
        self.img_item.setImage(img3d,levels=(0,255))

class DrawScale(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(DrawScale,self).__init__(mod_in,img_item,properties)
        self.properties['scale'] = 1

        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        # self.wImgBox_VB.setMouseEnabled(False,False)

        self.wPixels = QtGui.QLabel('1')
        self.wPixels.setFixedWidth(60)
        self.wScale = QtGui.QLabel('1')
        self.wScale.setFixedWidth(60)

        self.wLengthEdit = QtGui.QLineEdit(str(self.properties['scale']))
        self.wLengthEdit.setFixedWidth(60)
        self.wLengthEdit.setValidator(QtGui.QDoubleValidator())
        x,y = self.mod_in.image().shape
        self.roi = pg.LineSegmentROI([[int(x/2),int(y/4)],[int(x/2),int(3*y/4)]])
        self.wImgBox_VB.addItem(self.roi)

        self.wLayout.addWidget(QtGui.QLabel('# Pixels:'),0,0)
        self.wLayout.addWidget(self.wPixels,0,1)

        self.wLayout.addWidget(QtGui.QLabel('Length (um):'),1,0)
        self.wLayout.addWidget(self.wLengthEdit,1,1)

        self.wLayout.addWidget(QtGui.QLabel('Scale (um/px):'),2,0)
        self.wLayout.addWidget(self.wScale,2,1)

        self.wLayout.addWidget(self.wImgBox,3,0,4,4)

        self.roi.sigRegionChanged.connect(self.update_view)
        self.wLengthEdit.returnPressed.connect(self.update_view)
        self.wLengthEdit.textChanged.connect(self.update_view)

    def to_dict(self):
        d = super(DrawScale,self).to_dict()
        d['roi_state'] = self.roi.saveState()
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(DrawScale,cls).from_dict(d,img_item)
        obj.roi.setState(d['roi_state'])
        obj.wLengthEdit.setText(str(d['properties']['scale_length_um']))
        obj.update_image()
        obj.widget().hide()

        return obj

    def update_image(self):
        self.properties['num_pixels'] = len(self.roi.getArrayRegion(self.mod_in.image(),self.img_item))
        self.wPixels.setNum(self.properties['num_pixels'])
        self.properties['scale_length_um'] = float(self.wLengthEdit.text())
        if self.properties['num_pixels'] != 0:
            self.properties['scale'] = self.properties['scale_length_um'] / self.properties['num_pixels']
        self.wScale.setText(str(self.properties['scale']))

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Draw Scale'

class Erase(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(Erase,self).__init__(mod_in,img_item,properties)
        self.img_out = self.mod_in.image()
        self.eraser_size = 10
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_out,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)

        self.wSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(100)
        self.wSizeSlider.setSliderPosition(self.eraser_size)

        kern = (np.ones((self.eraser_size,self.eraser_size))*255).astype(np.uint8)
        self.wImgROI.setDrawKernel(kern, mask=None, center=(int(self.eraser_size/2),int(self.eraser_size/2)), mode='set')
        self.wSizeSlider.valueChanged.connect(self.update_view)

        self.wLayout.addWidget(QtGui.QLabel('Eraser Size:'),0,0)
        self.wLayout.addWidget(self.wSizeSlider,0,1)
        self.wLayout.addWidget(self.wImgBox,1,0,4,4)

    def to_dict(self):
        d = super(Erase,self).to_dict()
        d['eraser_size'] = self.eraser_size
        d['erased_image'] = self.img_out.tolist()
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(Erase,cls).from_dict(d,img_item)
        obj.eraser_size = d['eraser_size']
        obj.wSizeSlider.setSliderPosition(d['eraser_size'])
        obj.wImgROI.setImage(np.array(d['erased_image']),levels=(0,255))
        obj.update_image()

        obj.widget().hide()
        return obj

    def update_image(self):
        self.eraser_size = int(self.wSizeSlider.value())
        kern = (np.ones((self.eraser_size,self.eraser_size))*255).astype(np.uint8)
        self.wImgROI.setDrawKernel(kern, mask=None, center=(int(self.eraser_size/2),int(self.eraser_size/2)), mode='set')
        self.img_out = self.wImgROI.image

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Erase'

class SobelFilter(Modification):
    def __init__(self,mod_in,img_item,properties={}):
        super(SobelFilter,self).__init__(mod_in,img_item,properties)
        self.sobel_size = 3
        self.convolution = np.zeros(60)

        self._maskClasses = OrderedDict()
        self._maskClasses['Custom'] = CustomFilter
        self._maskClasses['Erase'] = EraseFilter

        self.wFilterType = QtWidgets.QComboBox()
        self.wFilterType.addItems(list(self._maskClasses.keys()))

        self.wFilterList = QtGui.QListWidget()
        self.wFilterList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.wAdd = QtGui.QPushButton('Add Region')
        self.wRemove = QtGui.QPushButton('Delete Region')

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = SmartImageItem()
        self.wImgROI.setImage(self.mod_in.image(),levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        self.wImgBox_VB.sigResized.connect(lambda v: self.wImgROI.updateCursor())
        self.wImgBox_VB.sigTransformChanged.connect(lambda v: self.wImgROI.updateCursor())

        layer_layout = QtGui.QGridLayout()
        layer_layout.setAlignment(QtCore.Qt.AlignTop)
        layer_layout.addWidget(self.wFilterList,0,0,6,1)
        layer_layout.addWidget(self.wFilterType,0,1)
        layer_layout.addWidget(self.wAdd,1,1)
        layer_layout.addWidget(self.wRemove,2,1)

        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QtCore.Qt.AlignTop)

        self.wSobelSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wSobelSizeSlider.setMinimum(1)
        self.wSobelSizeSlider.setMaximum(3)
        self.wSobelSizeSlider.setSliderPosition(2)

        self.wMinLengthSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wMinLengthSlider.setMinimum(3)
        self.wMinLengthSlider.setMaximum(10)
        self.wMinLengthSlider.setSliderPosition(3)

        self.wSNRSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wSNRSlider.setMinimum(1)
        self.wSNRSlider.setMaximum(100)
        self.wSNRSlider.setSliderPosition(20)

        self.wNoisePercSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.wNoisePercSlider.setMinimum(1)
        self.wNoisePercSlider.setMaximum(99)
        self.wNoisePercSlider.setSliderPosition(10)

        self.wHistPlot = pg.PlotWidget(title='Angle Histogram',pen=pg.mkPen(color='k',width=4))
        self.wHistPlot.setXRange(0,180)
        self.wHistPlot.hideAxis('left')
        self.wHistPlot.setLabel('bottom',text="Angle")

        self.wConvPlot = pg.PlotWidget(title='Convolution with Comb Function',pen=pg.mkPen(color='k',width=4))
        self.wConvPlot.setXRange(0,60)
        self.wConvPlot.hideAxis('left')
        self.wConvPlot.setLabel('bottom',text="Angle")

        self.wStd = QtGui.QLabel('')
        self.shiftAngle = QtGui.QLabel('')

        self.exportHistBtn = QtGui.QPushButton('Export Histogram')
        self.exportConvBtn = QtGui.QPushButton('Export Convolution')
        self.exportDataBtn = QtGui.QPushButton('Export Data')

        self.colors = QtGui.QComboBox()
        self.colors.addItems(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'])

        self.customShift = QtGui.QLineEdit('')

        self.wLayout.addWidget(QtGui.QLabel('Size:'),0,0)
        self.wLayout.addWidget(self.wSobelSizeSlider,0,1)

        self.wLayout.addWidget(self.wHistPlot,4,0,4,4)
        self.wLayout.addWidget(self.wConvPlot,8,0,4,4)
        self.wLayout.addWidget(QtGui.QLabel('Shifted St. Dev.:'),12,0)
        self.wLayout.addWidget(self.wStd,12,1)
        self.wLayout.addWidget(QtGui.QLabel("Color:"),13,0)
        self.wLayout.addWidget(self.colors,13,1)
        self.wLayout.addWidget(QtGui.QLabel("Custom Shift:"),14,0)
        self.wLayout.addWidget(QtGui.QLabel("Shift Angle:"),15,0)
        self.wLayout.addWidget(self.shiftAngle,15,1)
        self.wLayout.addWidget(self.customShift,14,1)
        self.wLayout.addWidget(self.exportHistBtn,18,0)
        self.wLayout.addWidget(self.exportConvBtn,18,1)
        self.wLayout.addWidget(self.exportDataBtn,19,0,2)

        self.wSobelSizeSlider.valueChanged.connect(self.update_view)
        self.exportHistBtn.clicked.connect(lambda: self.export(self.wHistPlot.getPlotItem()))
        self.exportConvBtn.clicked.connect(lambda: self.export(self.wConvPlot.getPlotItem()))
        self.exportDataBtn.clicked.connect(self.exportData)
        self.colors.currentIndexChanged.connect(lambda x: self.update_view())
        self.customShift.textChanged.connect(lambda x: self.update_view())

        self.update_view()

    @errorCheck(error_text="Error exporting data!")
    def exportData(self):
        path = os.path.join(os.getcwd(),"untitled.json")
        filename = QtWidgets.QFileDialog.getSaveFileName(None,
            "Export Data",
            path,
            "JSON (*.json)",
            "JSON (*.json)")[0]
        with open(filename,'w') as f:
            json.dump(self.properties,f)


    @errorCheck(error_text="Error exporting item!")
    def export(self,item):
        default_name = "untitled"
        exporter = pyqtgraph.exporters.ImageExporter(item)
        exporter.parameters()['width'] = 1024
        if self.properties['mode'] == 'local':
            path = os.path.join(os.getcwd(),default_name+".png")
            name = QtWidgets.QFileDialog.getSaveFileName(None, 
                "Export Image", 
                path, 
                "PNG File (*.png)",
                "PNG File (*.png)")[0]
            if name != '' and check_extension(name, [".png"]):
                exporter.export(fileName=name)
        elif self.properties["mode"] == 'nanohub':
            name = default_name+".png"
            exporter.export(fileName=name)
            subprocess.check_output('exportfile %s'%name,shell=True)
            try:
                os.remove(name)
            except:
                pass
        else:
            return

    def to_dict(self):
        d = super(SobelFilter,self).to_dict()
        d['sobel'] = {
            'ksize': self.sobel_size
        }
        d['size_tick'] = int(self.wSobelSizeSlider.value())
        return d

    @classmethod
    def from_dict(cls,d,img_item):
        obj = super(SobelFilter,cls).from_dict(d,img_item)
        obj.sobel_size = d['sobel']['ksize']

        obj.wSobelSizeSlider.setSliderPosition(d['size_tick'])
        obj.update_image()
        obj.widget().hide()

        return obj

    def update_image(self):
        self.sobel_size = 2*int(self.wSobelSizeSlider.value())+1

        self.dx = cv2.Sobel(self.mod_in.image(),ddepth=cv2.CV_64F,dx=1,dy=0,ksize=self.sobel_size)
        self.dy = cv2.Sobel(self.mod_in.image(),ddepth=cv2.CV_64F,dx=0,dy=1,ksize=self.sobel_size)

        self.properties['theta'] = np.arctan2(self.dy,self.dx)*180/np.pi
        self.properties['magnitude'] = np.sqrt(self.dx**2+self.dy**2)

        self.properties['angle_histogram'] = {}
        self.properties['angle_histogram']['y'],self.properties['angle_histogram']['x'] = np.histogram(
            self.properties['theta'].flatten(),
            weights=self.properties['magnitude'].flatten(),
            bins=np.linspace(0,180,180),
            density=True)

        comb = np.zeros(120)
        comb[0] = 1
        comb[60] = 1
        comb[-1] = 1
        self.convolution = sc.signal.convolve(self.properties['angle_histogram']['y'],comb,mode='valid')
        self.convolution = self.convolution/sum(self.convolution)
        cos = np.average(np.cos(np.arange(len(self.convolution))*2*np.pi/60),weights=self.convolution)
        sin = np.average(np.sin(np.arange(len(self.convolution))*2*np.pi/60),weights=self.convolution)
        if self.customShift.text()!='':
            self.periodic_mean = int('0'+self.customShift.text())
        else:
            self.periodic_mean = np.round((np.arctan2(-sin,-cos)+np.pi)*60/2/np.pi).astype(int)
        self.convolution = np.roll(self.convolution,30-self.periodic_mean)
        self.periodic_var = np.average((np.arange(len(self.convolution))-30)**2,weights=self.convolution)

        self.properties['convolution'] = self.convolution
        self.properties['periodic_var'] = self.periodic_var

    def update_view(self):
        self.update_image()
        color = pg.mkColor(self.colors.currentText())
        color.setAlpha(150)

        self.img_item.setImage(self.properties['magnitude'])
        self.wConvPlot.clear()
        self.wConvPlot.plot(
            range(len(self.convolution)+1),
            self.convolution,
            stepMode=True,
            fillLevel=0,
            brush=color,
            pen=pg.mkPen(color='k',width=4))
        self.wConvPlot.addLine(x=30)
        self.wConvPlot.addLine(x=30-np.sqrt(self.periodic_var),pen=pg.mkPen(dash=[3,5],width=4))
        self.wConvPlot.addLine(x=30+np.sqrt(self.periodic_var),pen=pg.mkPen(dash=[3,5],width=4))
        self.wHistPlot.clear()

        self.wHistPlot.plot(
            self.properties['angle_histogram']['x'],
            self.properties['angle_histogram']['y'],
            stepMode=True,
            fillLevel=0,
            brush=color,
            pen=pg.mkPen(color='k',width=4))
        self.wStd.setNum(np.sqrt(self.periodic_var))
        self.shiftAngle.setNum(self.periodic_mean)

        return self.properties

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Sobel Filter'

class SmartImageItem(pg.ImageItem):
    imageUpdateSignal = QtCore.pyqtSignal(object,object)
    imageFinishSignal = QtCore.pyqtSignal()
    def __init__(self,*args,**kwargs):
        super(SmartImageItem,self).__init__(*args,**kwargs)
        self.base_cursor = self.cursor()
        self.radius = None
        self.clickOnly = False

    def drawAt(self, pos, ev=None):
        pos = [int(pos.x()), int(pos.y())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]
        
        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))

        self.imageUpdateSignal.emit(ts,dk[ss])

    def resetCursor(self):
        self.setCursor(self.base_cursor)
        self.radius = None

    def updateCursor(self,radius=None):
        if radius:
            self.radius = radius
        if self.radius:
            radius = self.radius
            o = self.mapToDevice(QtCore.QPointF(0,0))
            x = self.mapToDevice(QtCore.QPointF(1,0))
            # d = max(1, int(1.0 / Point(x-o).length()))
            d = 1.0 / Point(x-o).length()
            radius = int(radius/d)
            pix = QtGui.QPixmap(4*radius+1,4*radius+1)
            pix.fill(QtCore.Qt.transparent)

            paint = QtGui.QPainter(pix)
            paint.setRenderHint(QtGui.QPainter.Antialiasing)
            pt = QtCore.QPointF(2*radius,2*radius)
            paint.setBrush(QtCore.Qt.transparent)
            paint.drawEllipse(pt,radius,radius)
            paint.end()
            
            self.setCursor(QtGui.QCursor(pix))

    def disconnect(self):
        sigs = [
            self.imageUpdateSignal,
            self.imageFinishSignal
            ]
        for sig in sigs:
            if self.receivers(sig)>0:
                sig.disconnect()

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None and not self.clickOnly:
            ev.accept()
            self.drawAt(ev.pos(), ev)
            if ev.isFinish():
                self.imageFinishSignal.emit()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawKernel is not None and ev.button() == QtCore.Qt.LeftButton:
            self.drawAt(ev.pos(), ev)

    def setClickOnly(self,flag):
        assert isinstance(flag,bool)
        self.clickOnly = flag


        
def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'local'
    if mode not in ['nanohub','local']:
        mode = 'local'
    app = QtGui.QApplication([])
    img_analyzer = GSAImage(mode=mode)
    img_analyzer.run()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

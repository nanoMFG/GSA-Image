from __future__ import division
import numpy as np
import scipy as sc
import cv2, sys, time, json, copy, subprocess, os
from skimage import transform
from skimage import util
from skimage import color
import functools
from skimage.draw import circle as skcircle
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW
from PyQt5 import QtCore as QC
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph import Point
from pyqtgraph import SignalProxy
from PIL import Image
from PIL.ImageQt import ImageQt
from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from util.util import errorCheck, mask_color_img, check_extension, ConfigParams
from util.gwidgets import GStackedWidget, ImageWidget, StandardDisplay
from util.io import IO
from io import BytesIO
import traceback

pg.setConfigOption('background', 'w')
pg.setConfigOption('imageAxisOrder', 'row-major')

testMode = True

## TODO: add toolbar and/or menu bar
class Main(QW.QMainWindow):
    def __init__(self,mode='local',*args,**kwargs):
        super(Main,self).__init__(*args,**kwargs)

        self.mainWidget = GSAImage(mode=mode)
        self.setCentralWidget(self.mainWidget)

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)

        importAction = QG.QAction("&Import Image",self)
        importAction.triggered.connect(self.mainWidget.importTrigger)
        exportAction = QG.QAction("&Export Image",self)
        exportAction.triggered.connect(self.mainWidget.exportTrigger)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(importAction)
        fileMenu.addAction(exportAction)

        helpAction = QG.QAction("&About",self)
        helpAction.triggered.connect(self.showAboutDialog)

        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(helpAction)

        self.show()

    def showAboutDialog(self):
        about_dialog = QW.QMessageBox(self)
        about_dialog.setText("About This Tool")
        about_dialog.setWindowModality(QC.Qt.WindowModal)

        about_text = """
        """

        about_dialog.setInformativeText(about_text)
        about_dialog.exec()


class GSAImage(QW.QWidget):
    def __init__(self,mode='local',parent=None):
        super(GSAImage,self).__init__(parent=parent)
        self.config = ConfigParams(mode=mode)
        self.workingDirectory = os.getcwd()

        if self.config.mode == 'nanohub':
            if 'TempGSA' not in os.listdir(self.workingDirectory):
                os.mkdir('TempGSA')
            self.tempdir = os.path.join(os.getcwd(),'TempGSA')
            os.chdir(self.tempdir)
            self.workingDirectory = os.path.join(self.workingDirectory,'TempGSA')

        self.mod_dict = {
        'Color Mask': ColorMask,
        'Canny Edge Detector': CannyEdgeDetection,
        'Dilate': Dilation,
        'Erode': Erosion,
        'Binary Mask': BinaryMask,
        'Filter Pattern': FilterPattern,
        'Blur': Blur,
        'Draw Scale': DrawScale,
        'Crop': Crop,
        'Domain Centers': DomainCenters,
        'Erase': Erase,
        'Sobel Filter': SobelFilter,
        'Remove Scale': RemoveScale
        }

        self.modComboBox = pg.ComboBox()
        for item in sorted(list(self.mod_dict)):
            self.modComboBox.addItem(item)

        font = QG.QFont()
        font.setPointSize(22)
        font.setWeight(QG.QFont.ExtraBold)

        self.addBtn = QG.QPushButton('+')
        self.addBtn.setFont(font)
        self.addBtn.setMaximumWidth(32)
        self.addBtn.setMaximumHeight(32)
        self.addBtn.clicked.connect(lambda: self.addMod(mod=None))

        self.removeBtn = QG.QPushButton('-')
        self.removeBtn.setFont(font)
        self.removeBtn.setMaximumWidth(32)
        self.removeBtn.setMaximumHeight(32)
        self.removeBtn.clicked.connect(self.removeMod)

        self.io_buttons = IO(
            config=self.config,
            imptext='Import Image',
            exptext='Export Image',
            ftype='image',
            extension='png')
        self.io_buttons.importClicked.connect(self.importImage)
        self.io_buttons.exportClicked.connect(lambda: self.io_buttons.exportData.emit(self.image()))

        self.stackedControl = GStackedWidget()
        self.mod_list = self.stackedControl.createListView()

        self.mod_list.setIconSize(QC.QSize(96, 96))

        self.stackedDisplay = QW.QStackedWidget()
        self.stackedControl.currentChanged.connect(lambda idx: self.stackedDisplay.setCurrentIndex(idx))
        self.stackedControl.widgetRemoved.connect(lambda idx:
            self.stackedDisplay.removeWidget(self.stackedDisplay.widget(idx)))
        self.stackedControl.currentChanged.connect(self.select)

        self.dummyControlWidget = QW.QWidget()
        self.toggleControl = QW.QStackedWidget()
        self.toggleControl.addWidget(self.dummyControlWidget)
        self.toggleControl.addWidget(self.stackedControl)

        self.mainImage = ImageWidget()
        self.toggleDisplay = QW.QStackedWidget()
        self.toggleDisplay.addWidget(self.mainImage)
        self.toggleDisplay.addWidget(self.stackedDisplay)

        layers = QG.QWidget()
        layers.setFixedWidth(250)
        mainLayout = QG.QGridLayout(layers)
        # mainLayout.addWidget(self.io_buttons, 0,0,1,3)
        mainLayout.addWidget(self.mod_list,2,0,1,3)
        mainLayout.addWidget(self.modComboBox,3,1)
        mainLayout.addWidget(self.addBtn,3,0)
        mainLayout.addWidget(self.removeBtn,3,2)

        self.layout = QG.QGridLayout(self)
        self.layout.addWidget(layers,0,0)
        self.layout.addWidget(self.toggleDisplay,0,1)
        self.layout.addWidget(self.toggleControl,0,2)

    def importTrigger(self):
        return self.io_buttons.importFile()

    def exportTrigger(self):
        self.io_buttons.requestExportData()

    # @errorCheck(skip=testMode)
    def importImage(self,filepath):
        self.clear()
        try:
            img = cv2.imread(filepath)
        except:
            raise IOError("Cannot read file %s"%os.path.basename(filepath))
        if isinstance(img,np.ndarray):
            mod = InitialImage()
            mod.set_image(img)
            self.setWindowTitle(os.path.basename(os.path.basename(filepath)))
            self.addMod(mod)

    @errorCheck(skip=testMode)
    def image(self,index=None):
        count = self.stackedControl.count()
        if count > 0:
            if index is None:
                index = count-1
            return self.stackedControl[index].image()
        else:
            raise IOError("No image to export.")

    def clear(self):
        self.stackedControl.clear()

    def removeMod(self):
        if self.stackedControl.count()>0:
            self.stackedControl.removeIndex(self.stackedControl.count()-1)

    @errorCheck(error_text='Error adding layer!',skip=testMode)
    def addMod(self,mod=None):
        if mod is None:
            if self.stackedControl.count() > 0:
                mod = self.mod_dict[self.modComboBox.value()](
                    self.stackedControl[self.stackedControl.count()-1],
                    properties={'mode':self.config.mode})
            else:
                raise ValueError("You need to import an image before adding layers.")
        self.stackedControl.addWidget(
            mod,
            name=mod.name(),
            icon=mod.icon())
        self.stackedDisplay.addWidget(mod.display())

        index = self.stackedControl.count()-1
        self.stackedControl.setCurrentIndex(index)

        mod.imageChanged.connect(lambda image: self.stackedControl.setIcon(index,mod.icon()))
        mod.emitImage()

    def select(self, index):
        if index == self.stackedControl.count()-1:
            self.toggleDisplay.setCurrentIndex(1)
            self.toggleControl.setCurrentIndex(1)
        else:
            self.toggleDisplay.setCurrentIndex(0)
            self.toggleControl.setCurrentIndex(0)
            self.mainImage.setImage(self.stackedControl[index].image(),levels=(0,255))
        if index >= 0:
            self.stackedControl[index].update_view()

    def widget(self):
        return self

    def run(self):
        self.show()


class Modification(QW.QScrollArea):
    """
    Abstract class for defining modifications to an image. Modifications form a linked list with each object
    inheriting an input Modification. In this way, images are modified with sequential changes. Modification
    objects should also connect to GSAImage's ImageItem, which contains the main image display so that item
    can be updated.

    inputMod:         the Modification that the current object inherits.
    properties:     a dictionary of properties that may be used for the modification.
    """

    imageChanged = QC.pyqtSignal(object)
    def __init__(self,inputMod=None,properties={},parent=None):
        super(Modification,self).__init__(parent=parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        self.inputMod = inputMod
        self.properties = properties

        self._display = ImageWidget()
        self.imageChanged.connect(lambda image: self._display.setImage(image,levels=(0,255)))
        if inputMod is not None:
            self.set_image(self.inputMod.image())
        else:
            self.img_out = None

    def display(self):
        return self._display

    def emitImage(self):
        self.imageChanged.emit(self.image())

    def icon(self):
        if self.img_out is not None:
            pic = Image.fromarray(self.img_out)
            # size = self.mod_list.iconSize()
            # pic.thumbnail((size.width(),size.height()))

            data = pic.convert("RGB").tobytes("raw","RGB")

            img = QG.QImage(data, pic.size[0], pic.size[1], QG.QImage.Format_RGB888)
            pix = QG.QPixmap.fromImage(img)
            icon = QG.QIcon(pix)
            icon.addPixmap(pix,QG.QIcon.Selected)

            return icon
        return QG.QIcon()

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
        self.imageChanged.emit(self.img_out)
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
        return self.properties
    def delete_mod(self):
        """
        Deletes the modification and returns the inherited modification.
        """
        return self.inputMod
    def tolist(self):
        """
        Converts linked list to list.
        """
        if self.inputMod != None:
            return self.inputMod.tolist() + [self]
        else:
            return [self]
    def back_traverse(self,n):
        """
        Gives the modification that is n modifications back from the end of the list.

        n:              number of indices to traverse backwards on linked list.
        """
        if n != 0:
            if self.inputMod == None:
                raise IndexError('Index out of range (n = %d)'%n)
            elif n != 0:
                return self.inputMod.back_traverse(n-1)
        elif n == 0:
            return self
    def root(self):
        """
        Gives the first Modification in linked list.
        """
        if self.inputMod != None:
            return self.inputMod.root()
        else:
            return self
    def length(self):
        """
        Gives the length of the linked list.
        """
        if self.inputMod != None:
            return self.inputMod.length()+1
        else:
            return 1
    def back_properties(self):
        """
        Gives a dictionary containing all properties of current and inherited Modifications.
        """
        if self.inputMod != None:
            d = self.inputMod.back_properties()
            d.update(self.properties)
            return d
        else:
            d = {}
            d.update(self.properties)
            return d

class InitialImage(Modification):
    def __init__(self,*args,**kwargs):
        super(InitialImage,self).__init__(*args,**kwargs)

    def name(self):
        return 'Initial Image'

class RemoveScale(Modification):
    def __init__(self,inputMod,properties={}):
        super(RemoveScale,self).__init__(inputMod,properties)

    def name(self):
        return 'Remove Scale'

    def crop_image(self,img,box):
        pil_img = Image.fromarray(img)
        return np.array(pil_img.crop(box))

    def update_image(self,scale_location='Auto',tol=0.95):
        img_array = self.inputMod.image()
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
            self.img_out = self.inputMod.image()


class ColorMask(Modification):
    def __init__(self,inputMod,properties={}):
        super(ColorMask,self).__init__(inputMod,properties)
        self.img_mask = None
        self.img_item = pg.ImageItem(self.inputMod.image())
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

        self.control = QW.QWidget()

    def widget(self):
        return self.wHistPlot

    def update_image(self):
        minVal, maxVal = self.lrItem.getRegion()
        img = self.inputMod.image()
        self.img_mask = np.zeros_like(img)
        self.img_mask[np.logical_and(img>minVal,img<maxVal)] = 1
        self.img_out = img*self.img_mask+(1-self.img_mask)*255

    def name(self):
        return 'Color Mask'

class CannyEdgeDetection(Modification):
    def __init__(self,inputMod,properties={}):
        super(CannyEdgeDetection,self).__init__(inputMod,properties)

        self.low_thresh = int(max(self.inputMod.image().flatten())*.1)
        self.high_thresh = int(max(self.inputMod.image().flatten())*.4)
        self.gauss_size = 5
        self.control = pg.LayoutWidget()
        self.control.layout.setAlignment(QC.Qt.AlignTop)

        self.wGaussEdit = QG.QLineEdit(str(self.gauss_size))
        self.wGaussEdit.setValidator(QG.QIntValidator(3,51))
        self.wGaussEdit.setFixedWidth(60)

        self.wLowSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wLowSlider.setMinimum(0)
        self.wLowSlider.setMaximum(255)
        self.wLowSlider.setSliderPosition(int(self.low_thresh))
        self.wLowEdit = QG.QLineEdit(str(self.low_thresh))
        self.wLowEdit.setFixedWidth(60)
        self.wLowEdit.setValidator(QG.QIntValidator(0,255))

        self.wHighSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wHighSlider.setMinimum(0)
        self.wHighSlider.setMaximum(255)
        self.wHighSlider.setSliderPosition(int(self.high_thresh))
        self.wHighEdit = QG.QLineEdit(str(self.high_thresh))
        self.wHighEdit.setFixedWidth(60)
        self.wHighEdit.setValidator(QG.QIntValidator(0,255))

        self.wGaussEdit.returnPressed.connect(self._update_sliders)
        self.wLowSlider.sliderReleased.connect(self._update_texts)
        self.wLowSlider.sliderMoved.connect(self._update_texts)
        self.wLowEdit.returnPressed.connect(self._update_sliders)
        self.wHighSlider.sliderReleased.connect(self._update_texts)
        self.wHighSlider.sliderMoved.connect(self._update_texts)
        self.wHighEdit.returnPressed.connect(self._update_sliders)

        self.control.addWidget(QG.QLabel('Gaussian Size'),0,0)
        self.control.addWidget(QG.QLabel('Low Threshold'),1,0)
        self.control.addWidget(QG.QLabel('High Threshold'),3,0)
        self.control.addWidget(self.wGaussEdit,0,1)
        self.control.addWidget(self.wLowEdit,1,1)
        self.control.addWidget(self.wHighEdit,3,1)
        self.control.addWidget(self.wLowSlider,2,0,1,2)
        self.control.addWidget(self.wHighSlider,4,0,1,2)

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
        self.img_out = cv2.GaussianBlur(self.inputMod.image(),(self.gauss_size,self.gauss_size),0)
        self.img_out = 255-cv2.Canny(self.img_out,self.low_thresh,self.high_thresh,L2gradient=True)

    def widget(self):
        return self.control

class Dilation(Modification):
    def __init__(self,inputMod,properties={}):
        super(Dilation,self).__init__(inputMod,properties)
        self.size = 1
        self.control = pg.LayoutWidget()
        self.control.layout.setAlignment(QC.Qt.AlignTop)
        self.wSizeEdit = QG.QLineEdit(str(self.size))
        self.wSizeEdit.setValidator(QG.QIntValidator(1,20))
        self.wSizeEdit.setFixedWidth(60)
        self.wSizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(20)
        self.wSizeSlider.setSliderPosition(int(self.size))

        # self.wSizeSlider.sliderReleased.connect(self._update_texts)
        self.wSizeSlider.valueChanged.connect(self._update_texts)
        self.wSizeEdit.returnPressed.connect(self._update_sliders)

        self.control.addWidget(QG.QLabel('Kernel Size'),0,0)
        self.control.addWidget(self.wSizeEdit,0,1)
        self.control.addWidget(self.wSizeSlider,1,0,1,2)

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
        self.img_out = cv2.erode(self.inputMod.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

    def widget(self):
        return self.control

class Erosion(Modification):
    def __init__(self,inputMod,properties={}):
        super(Erosion,self).__init__(inputMod,properties)
        self.size = 1
        self.control = pg.LayoutWidget()
        self.control.layout.setAlignment(QC.Qt.AlignTop)
        self.wSizeEdit = QG.QLineEdit(str(self.size))
        self.wSizeEdit.setValidator(QG.QIntValidator(1,20))
        self.wSizeEdit.setFixedWidth(60)
        self.wSizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(20)
        self.wSizeSlider.setSliderPosition(int(self.size))

        # self.wSizeSlider.sliderReleased.connect(self._update_texts)
        self.wSizeSlider.valueChanged.connect(self._update_texts)
        self.wSizeEdit.returnPressed.connect(self._update_sliders)

        self.control.addWidget(QG.QLabel('Kernel Size'),0,0)
        self.control.addWidget(self.wSizeEdit,0,1)
        self.control.addWidget(self.wSizeSlider,1,0,1,2)

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
        self.img_out = cv2.dilate(self.inputMod.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

    def widget(self):
        return self.control

class BinaryMask(Modification):
    def __init__(self,inputMod,properties={}):
        super(BinaryMask,self).__init__(inputMod,properties)

    def name(self):
        return 'Binary Mask'

    def update_image(self):
        self.img_out = self.inputMod.image()
        self.img_out[self.img_out < 255] = 0

class Blur(Modification):
    def __init__(self,inputMod,properties={}):
        super(Blur,self).__init__(inputMod,properties)
        self.gauss_size = 5
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QC.Qt.AlignTop)
        self.wGaussEdit = QG.QLineEdit(str(self.gauss_size))
        self.wGaussEdit.setFixedWidth(100)
        self.wGaussEdit.setValidator(QG.QIntValidator(3,51))

        self.wLayout.addWidget(QG.QLabel('Gaussian Size:'),0,0)
        self.wLayout.addWidget(self.wGaussEdit,0,1)

        self.update_view()
        self.wGaussEdit.returnPressed.connect(self.update_view)

    def update_image(self):
        self.gauss_size = int(self.wGaussEdit.text())
        self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
        self.wGaussEdit.setText(str(self.gauss_size))
        self.img_out = cv2.GaussianBlur(self.inputMod.image(),(self.gauss_size,self.gauss_size),0)

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Blur'

class TemplateMatchingWidget(Modification):
    imageChanged = QC.pyqtSignal(object)
    def __init__(self,inputMod,img_item,vbox,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,inputMod,properties,parent=parent)
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.inputMod.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.inputMod.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in3d.copy()

        self.threshSlider = QG.QSlider(QC.Qt.Horizontal)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(1000)
        self.threshSlider.setSliderPosition(100)
        self.threshSlider.valueChanged.connect(self.update_view)

        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
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

        main_widget = QW.QWidget()
        layout = QG.QGridLayout(self)
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(QW.QLabel("Threshold:"),0,0)
        layout.addWidget(self.threshSlider,0,1)
        layout.addWidget(QW.QLabel("Template Size:"),1,0)
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
    imageChanged = QC.pyqtSignal(object)
    def __init__(self,inputMod,img_item,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,inputMod,img_item,properties,parent=parent)
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.inputMod.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.inputMod.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in3d.copy()

        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(2)
        self.sizeSlider.setMaximum(100)
        self.sizeSlider.setSliderPosition(20)
        self.sizeSlider.valueChanged.connect(self.updateKernelCursor)

        main_widget = QW.QWidget()
        layout = QG.QGridLayout(self)
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(QW.QLabel("Cursor Size:"),0,0)
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
        self._mask = np.ones_like(self.inputMod.image(),dtype=bool)

    def update_view(self,slices=None,mask=None):
        CustomFilter.update_view(self,slices,mask,comparator=np.logical_and)

    def updateKernelCursor(self,radius):
        CustomFilter.updateKernelCursor(self,radius,kern_val=0)

class ClusterFilter(Modification):
    imageChanged = QC.pyqtSignal(object)
    def __init__(self,inputMod,img_item,mask_in=None,img_in=None,properties={},parent=None):
        Modification.__init__(self,inputMod,img_item,properties,parent=parent)
        self._clusters = None
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.inputMod.image(),dtype=bool)
        if isinstance(img_in,np.ndarray):
            self.img_in = img_in
        else:
            self.img_in = self.inputMod.image()
        self.img_in3d = np.dstack((self.img_in, self.img_in, self.img_in))
        self.roi_img = self.img_in.copy()

        self.cluster_list = QW.QListWidget()
        self.cluster_list.setSelectionMode(QG.QAbstractItemView.MultiSelection)
        self.wsize_edit = QW.QLineEdit()
        self.wsize_edit.setValidator(QG.QIntValidator(1,50))
        self.wsize_edit.setText("15")
        self.run_btn = QW.QPushButton("Run")

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
            self._mask[self._clusters==item.data(QC.Qt.UserRole)] = True
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
                item = QG.QListWidgetItem(str(fraction))
                item.setData(QC.Qt.UserRole,label)
                self.cluster_list.addItem(item)

    def filter(self):
        pass

class KMeansFilter(ClusterFilter):
    def __init__(self,*args,**kwargs):
        ClusterFilter.__init__(self,*args,**kwargs)
        self.n_clusters_edit = QW.QLineEdit()
        self.n_clusters_edit.setValidator(QG.QIntValidator(2,20))
        self.n_clusters_edit.setText("2")

        self.stride_edit = QW.QLineEdit()
        self.stride_edit.setValidator(QG.QIntValidator(1,30))
        self.stride_edit.setText("3")

        main_widget = QW.QWidget()
        layout = QG.QGridLayout(self)
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(QW.QLabel("Window Size:"),0,0)
        layout.addWidget(self.wsize_edit,0,1)
        layout.addWidget(QW.QLabel("# Clusters:"),1,0)
        layout.addWidget(self.n_clusters_edit,1,1)
        layout.addWidget(QW.QLabel("Stride:"),2,0)
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
        self.n_components_edit = QW.QLineEdit()
        self.n_components_edit.setValidator(QG.QIntValidator(2,20))
        self.n_components_edit.setText("2")

        main_widget = QW.QWidget()
        layout = QG.QGridLayout(self)
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(QW.QLabel("Window Size:"),0,0)
        layout.addWidget(self.wsize_edit,0,1)
        layout.addWidget(QW.QLabel("# Components:"),1,0)
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
    def __init__(self,inputMod,img_item,properties={}):
        super(FilterPattern,self).__init__(inputMod,img_item,properties)

        self._maskClasses = OrderedDict()
        self._maskClasses['Template Match'] = TemplateMatchingWidget
        self._maskClasses['Gaussian Mixture'] = GMMFilter
        self._maskClasses['K-Means'] = KMeansFilter
        self._maskClasses['Custom'] = CustomFilter
        self._maskClasses['Erase'] = EraseFilter

        self.wFilterType = QW.QComboBox()
        self.wFilterType.addItems(list(self._maskClasses.keys()))

        self.wFilterList = QG.QListWidget()
        self.wFilterList.setSelectionMode(QG.QAbstractItemView.SingleSelection)
        self.wAdd = QG.QPushButton('Add Filter')
        self.wRemove = QG.QPushButton('Remove Layer')
        self.wExportMask = QG.QPushButton('Export Mask')

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = SmartImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        self.wImgBox_VB.sigResized.connect(lambda v: self.wImgROI.updateCursor())
        self.wImgBox_VB.sigTransformChanged.connect(lambda v: self.wImgROI.updateCursor())
        
        self.blankWidget = QW.QWidget()
        self.stackedControl = QW.QStackedWidget()

        self.toggleControl = QW.QStackedWidget()
        self.toggleControl.addWidget(self.blankWidget)
        self.toggleControl.addWidget(self.stackedControl)

        layer_layout = QG.QGridLayout()
        layer_layout.setAlignment(QC.Qt.AlignTop)
        layer_layout.addWidget(self.wFilterList,0,0,6,1)
        layer_layout.addWidget(self.wFilterType,0,1)
        layer_layout.addWidget(self.wAdd,1,1)
        layer_layout.addWidget(self.wRemove,2,1)
        layer_layout.addWidget(self.wExportMask,4,1)
        
        main_widget = QW.QWidget()
        main_layout = QG.QGridLayout(main_widget)
        main_layout.setAlignment(QC.Qt.AlignTop)

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
        if isinstance(value,QG.QListWidgetItem):
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
            in_mod = self.inputMod
            mask_in = None

        if method == 'Template Match':
            mask_widget = maskClass(inputMod=in_mod,img_item=self.wImgROI,vbox=self.wImgBox_VB,mask_in=mask_in,img_in=self.inputMod.image())
        else:
            mask_widget = maskClass(inputMod=in_mod,img_item=self.wImgROI,mask_in=mask_in,img_in=self.inputMod.image())

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
            name = QW.QFileDialog.getSaveFileName(None, 
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
    def __init__(self,inputMod,img_item,properties={}):
        super(Crop,self).__init__(inputMod,img_item,properties)
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QC.Qt.AlignTop)

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

    def update_image(self):
        self.img_out,coords = self.roi.getArrayRegion(self.wImgROI.image,self.wImgROI,returnMappedCoords=True)
        self.img_out = self.img_out.astype(np.uint8)

        self.properties['crop_coords'] = coords.tolist()

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Crop'

class DomainCenters(Modification):
    def __init__(self,inputMod,img_item,properties={}):
        super(DomainCenters,self).__init__(inputMod,img_item,properties)
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
        self.img_out = self.inputMod.image()

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = SmartImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)

        self.wImgROI.setImage(self.img_out,levels=(0,255))
        self.wImgROI.setClickOnly(True)
        self.updateKernel(self.radius)

        self.domain_list = QW.QListWidget()
        self.deleteBtn = QW.QPushButton("Delete")
        self.exportBtn = QW.QPushButton("Export")

        main_widget = QW.QWidget()

        layer_layout = QG.QGridLayout(main_widget)
        layer_layout.setAlignment(QC.Qt.AlignTop)
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
            name = QW.QFileDialog.getSaveFileName(None, 
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
    def __init__(self,inputMod,img_item,properties={}):
        super(DrawScale,self).__init__(inputMod,img_item,properties)
        self.properties['scale'] = 1

        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QC.Qt.AlignTop)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_item.image,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)
        # self.wImgBox_VB.setMouseEnabled(False,False)

        self.wPixels = QG.QLabel('1')
        self.wPixels.setFixedWidth(60)
        self.wScale = QG.QLabel('1')
        self.wScale.setFixedWidth(60)

        self.wLengthEdit = QG.QLineEdit(str(self.properties['scale']))
        self.wLengthEdit.setFixedWidth(60)
        self.wLengthEdit.setValidator(QG.QDoubleValidator())
        x,y = self.inputMod.image().shape
        self.roi = pg.LineSegmentROI([[int(x/2),int(y/4)],[int(x/2),int(3*y/4)]])
        self.wImgBox_VB.addItem(self.roi)

        self.wLayout.addWidget(QG.QLabel('# Pixels:'),0,0)
        self.wLayout.addWidget(self.wPixels,0,1)

        self.wLayout.addWidget(QG.QLabel('Length (um):'),1,0)
        self.wLayout.addWidget(self.wLengthEdit,1,1)

        self.wLayout.addWidget(QG.QLabel('Scale (um/px):'),2,0)
        self.wLayout.addWidget(self.wScale,2,1)

        self.wLayout.addWidget(self.wImgBox,3,0,4,4)

        self.roi.sigRegionChanged.connect(self.update_view)
        self.wLengthEdit.returnPressed.connect(self.update_view)
        self.wLengthEdit.textChanged.connect(self.update_view)

    def update_image(self):
        self.properties['num_pixels'] = len(self.roi.getArrayRegion(self.inputMod.image(),self.img_item))
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
    def __init__(self,inputMod,img_item,properties={}):
        super(Erase,self).__init__(inputMod,img_item,properties)
        self.img_out = self.inputMod.image()
        self.eraser_size = 10
        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QC.Qt.AlignTop)

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_out,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)

        self.wSizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wSizeSlider.setMinimum(1)
        self.wSizeSlider.setMaximum(100)
        self.wSizeSlider.setSliderPosition(self.eraser_size)

        kern = (np.ones((self.eraser_size,self.eraser_size))*255).astype(np.uint8)
        self.wImgROI.setDrawKernel(kern, mask=None, center=(int(self.eraser_size/2),int(self.eraser_size/2)), mode='set')
        self.wSizeSlider.valueChanged.connect(self.update_view)

        self.wLayout.addWidget(QG.QLabel('Eraser Size:'),0,0)
        self.wLayout.addWidget(self.wSizeSlider,0,1)
        self.wLayout.addWidget(self.wImgBox,1,0,4,4)

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
    def __init__(self,inputMod,img_item,properties={}):
        super(SobelFilter,self).__init__(inputMod,img_item,properties)
        self.sobel_size = 3
        self.convolution = np.zeros(60)

        self.wLayout = pg.LayoutWidget()
        self.wLayout.layout.setAlignment(QC.Qt.AlignTop)

        self.wSobelSizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.wSobelSizeSlider.setMinimum(1)
        self.wSobelSizeSlider.setMaximum(3)
        self.wSobelSizeSlider.setSliderPosition(2)

        self.wHistPlot = pg.PlotWidget(title='Angle Histogram',pen=pg.mkPen(color='k',width=4))
        self.wHistPlot.setXRange(0,180)
        self.wHistPlot.hideAxis('left')

        self.wConvPlot = pg.PlotWidget(title='Convolution with Comb Function',pen=pg.mkPen(color='k',width=4))
        self.wConvPlot.setXRange(0,60)
        self.wConvPlot.hideAxis('left')

        self.wStd = QG.QLabel('')

        self.exportHistBtn = QG.QPushButton('Export Histogram')
        self.exportConvBtn = QG.QPushButton('Export Convolution')

        self.wLayout.addWidget(QG.QLabel('Size:'),0,0)
        self.wLayout.addWidget(self.wSobelSizeSlider,0,1)

        self.wLayout.addWidget(self.wHistPlot,4,0,4,4)
        self.wLayout.addWidget(self.wConvPlot,8,0,4,4)
        self.wLayout.addWidget(QG.QLabel('Shifted St. Dev.:'),12,0)
        self.wLayout.addWidget(self.wStd,12,1)
        self.wLayout.addWidget(self.exportHistBtn,13,0)
        self.wLayout.addWidget(self.exportConvBtn,13,1)

        self.wSobelSizeSlider.valueChanged.connect(self.update_view)

        self.update_view()
        self.exportHistBtn.clicked.connect(lambda: self.export(self.wHistPlot.getPlotItem()))
        self.exportConvBtn.clicked.connect(lambda: self.export(self.wConvPlot.getPlotItem()))

    @errorCheck(error_text="Error exporting item!")
    def export(self,item):
        default_name = "untitled"
        exporter = pyqtgraph.exporters.ImageExporter(item)
        exporter.parameters()['width'] = 1024
        if self.properties['mode'] == 'local':
            path = os.path.join(os.getcwd(),default_name+".png")
            name = QW.QFileDialog.getSaveFileName(None, 
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

    def update_image(self):
        self.sobel_size = 2*int(self.wSobelSizeSlider.value())+1

        self.dx = cv2.Sobel(self.inputMod.image(),ddepth=cv2.CV_64F,dx=1,dy=0,ksize=self.sobel_size)
        self.dy = cv2.Sobel(self.inputMod.image(),ddepth=cv2.CV_64F,dx=0,dy=1,ksize=self.sobel_size)

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
        self.periodic_mean = np.round((np.arctan2(-sin,-cos)+np.pi)*60/2/np.pi).astype(int)
        self.convolution = np.roll(self.convolution,30-self.periodic_mean)
        self.periodic_var = np.average((np.arange(len(self.convolution))-30)**2,weights=self.convolution)

        self.properties['convolution'] = self.convolution
        self.properties['periodic_var'] = self.periodic_var

    def update_view(self):
        self.update_image()
        self.img_item.setImage(self.properties['magnitude'])
        self.wConvPlot.clear()
        self.wConvPlot.plot(
            range(len(self.convolution)+1),
            self.convolution,
            stepMode=True,
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
            brush=(0,0,255,150),
            pen=pg.mkPen(color='k',width=4))
        self.wStd.setNum(np.sqrt(self.periodic_var))

        return self.properties

    def widget(self):
        return self.wLayout

    def name(self):
        return 'Sobel Filter'

class SmartImageItem(pg.ImageItem):
    imageUpdateSignal = QC.pyqtSignal(object,object)
    imageFinishSignal = QC.pyqtSignal()
    def __init__(self,*args,**kwargs):
        super(SmartImageItem,self).__init__(*args,**kwargs)
        self.base_cursor = self.cursor()
        self.radius = None
        self.clickOnly = False

    def setImage(self,img,*args,**kwargs):
        super(SmartImageItem,self).setImage(img.T,*args,**kwargs)

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
            o = self.mapToDevice(QC.QPointF(0,0))
            x = self.mapToDevice(QC.QPointF(1,0))
            # d = max(1, int(1.0 / Point(x-o).length()))
            d = 1.0 / Point(x-o).length()
            radius = int(radius/d)
            pix = QG.QPixmap(4*radius+1,4*radius+1)
            pix.fill(QC.Qt.transparent)

            paint = QG.QPainter(pix)
            paint.setRenderHint(QG.QPainter.Antialiasing)
            pt = QC.QPointF(2*radius,2*radius)
            paint.setBrush(QC.Qt.transparent)
            paint.drawEllipse(pt,radius,radius)
            paint.end()
            
            self.setCursor(QG.QCursor(pix))

    def disconnect(self):
        sigs = [
            self.imageUpdateSignal,
            self.imageFinishSignal
            ]
        for sig in sigs:
            if self.receivers(sig)>0:
                sig.disconnect()

    def mouseDragEvent(self, ev):
        if ev.button() != QC.Qt.LeftButton:
            ev.ignore()
            return
        elif self.drawKernel is not None and not self.clickOnly:
            ev.accept()
            self.drawAt(ev.pos(), ev)
            if ev.isFinish():
                self.imageFinishSignal.emit()

    def mouseClickEvent(self, ev):
        if ev.button() == QC.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawKernel is not None and ev.button() == QC.Qt.LeftButton:
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
    app = QG.QApplication([])
    # img_analyzer = GSAImage(mode=mode)
    # img_analyzer.run()
    main = Main()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

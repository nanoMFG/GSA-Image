from __future__ import division

from collections import OrderedDict

import cv2
import json
import numpy as np
import numpy.linalg as la
import os
import pyqtgraph as pg
import pyqtgraph.exporters
from scipy import signal
import seaborn
import subprocess
import sys
from PIL import Image
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from skimage import util
from skimage.draw import circle as skcircle
from skimage.draw import line
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from util.gwidgets import *
from util.icons import Icon
from util.io import IO
from util.util import errorCheck, mask_color_img, check_extension, ConfigParams
# from keras.models import load_model

pg.setConfigOption('background', 'w')
pg.setConfigOption('imageAxisOrder', 'row-major')

QW=QtWidgets
QC=QtCore
QG=QtGui

UNET_MODEL_PATH = ""

class Main(QW.QMainWindow):
    """
    Main window containing the GSAImage widget. Adds menu bar / tool bar functionality.
    """
    def __init__(self,mode='local', repo_dir = '', *args,**kwargs):
        super(Main,self).__init__(*args,**kwargs)

        self.mode = mode
        self.repo_dir = repo_dir
        self.mainWidget = GSAImage(mode=mode)
        self.setCentralWidget(self.mainWidget)

        # building main menu
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)

        importAction = QG.QAction("&Import",self)
        importAction.setIcon(Icon('download.svg'))
        importAction.triggered.connect(self.mainWidget.importTrigger)

        exportAction = QG.QAction("&Export",self)
        exportAction.setIcon(Icon('upload.svg'))
        exportAction.triggered.connect(self.mainWidget.exportTrigger)

        clearAction = QG.QAction("&Clear",self)
        clearAction.setIcon(Icon('trash.svg'))
        clearAction.triggered.connect(lambda _: self.mainWidget.clear())

        exitAction = QG.QAction("&Exit",self)
        exitAction.setIcon(Icon('log-out.svg'))
        exitAction.triggered.connect(self.close)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(importAction)
        fileMenu.addAction(exportAction)
        fileMenu.addAction(clearAction)
        if mode == 'local':
            fileMenu.addAction(exitAction)

        aboutAction = QG.QAction("&About",self)
        aboutAction.setIcon(Icon('info.svg'))
        aboutAction.triggered.connect(self.showAboutDialog)

        testImageAction = QG.QAction("&Import Test Image",self)
        testImageAction.setIcon(Icon('image.svg'))
        testImageAction.triggered.connect(self.importTestImage)

        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(testImageAction)
        helpMenu.addAction(aboutAction)

        self.show()

    def showAboutDialog(self):
        about_dialog = QW.QMessageBox(self)
        about_dialog.setText("About This Tool")
        about_dialog.setWindowModality(QC.Qt.WindowModal)
        copyright_path = os.path.join(self.repo_dir,'COPYRIGHT')
        print(f"okay:{copyright_path}")
        if os.path.isfile(copyright_path):
            with open(copyright_path,'r') as f:
                copyright = f.read()
                print(f"hey:{copyright}")
        else:
            copyright = ""

        version_path =  os.path.join(self.repo_dir,'VERSION')
        if os.path.isfile(version_path):
            with open(os.path.join(self.repo_dir,'VERSION'),'r') as f:
                version = f.read()
        else:
            version = ""

        # Needs text
        about_text = "Version: %s \n\n"%version
        about_text += copyright

        about_dialog.setInformativeText(about_text)
        about_dialog.exec()

    def importTestImage(self):
        path = os.path.join(self.repo_dir,'data','test.tif')
        self.mainWidget.importImage(path)

class GSAImage(QW.QWidget):
    def __init__(self,mode='local',parent=None):
        super(GSAImage,self).__init__(parent=parent)
        self.config = ConfigParams(mode=mode)
        self.workingDirectory = os.getcwd()
        self.controlWidth = 275

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
        'Alignment': Alignment,
        'Remove Scale': RemoveScale
        }

        self.modComboBox = pg.ComboBox()
        for item in sorted(list(self.mod_dict)):
            self.modComboBox.addItem(item)

        # Add / remove modification buttons
        self.addBtn = QG.QPushButton()
        self.addBtn.setIcon(Icon("plus.svg"))
        self.addBtn.setMaximumWidth(48)
        self.addBtn.clicked.connect(lambda: self.addMod(mod=None))

        self.removeBtn = QG.QPushButton()
        self.removeBtn.setIcon(Icon("minus.svg"))
        self.removeBtn.setMaximumWidth(48)
        self.removeBtn.clicked.connect(self.removeMod)

        # I/O buttons not used bc of file menu. However the class is used for facilitating import/export
        self.io_buttons = IO(
            config=self.config,
            imptext='Import Image',
            exptext='Export Image',
            ftype='image',
            extension='png')

        # stackedControl contains the modification widget
        self.stackedControl = GStackedWidget()
        # mod_list is the sidebar list w/ thumbnails
        self.mod_list = self.stackedControl.createListView()
        self.mod_list.setIconSize(QC.QSize(96, 96))

        # stackedDisplay holds the displays for each widget (Modification.display())
        self.stackedDisplay = QW.QStackedWidget()
        self.stackedDisplay.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)

        # defaultDisplay shown if selection is not last item in stack (to prevent editing prior layers)
        # defaultDisplay is set to the image from a particular selected layer in GSAImage.select
        self.defaultDisplay = ImageWidget()
        self.defaultDisplay.setMinimumWidth(int(self.defaultDisplay.sizeHint().width()*.75))
        self.controlDisplay = ControlImageWidget(self.defaultDisplay)

        # dummyControlWidget shown if selection is not last item in stack (to prevent editing prior layers)
        self.dummyControlWidget = QW.QWidget()
        self.toggleControl = QW.QStackedWidget()
        self.toggleControl.addWidget(self.dummyControlWidget)
        self.toggleControl.addWidget(self.stackedControl)

        self.toggleDisplay = QW.QStackedWidget()
        self.toggleDisplay.addWidget(self.defaultDisplay)
        self.toggleDisplay.addWidget(self.stackedDisplay)

        self.mod_list.setFixedWidth(250)

        bar_layout = QG.QGridLayout()
        bar_layout.addWidget(self.modComboBox,0,0)
        bar_layout.addWidget(self.addBtn,0,1)
        bar_layout.addWidget(self.removeBtn,0,2)

        layerLayout = QG.QGridLayout()
        layerLayout.addWidget(HeaderLabel('Layers'),0,0)
        layerLayout.addWidget(self.mod_list,2,0,1,3)
        layerLayout.addWidget(self.modComboBox,3,0)
        layerLayout.addWidget(self.addBtn,3,2)
        layerLayout.addWidget(self.removeBtn,3,1)
        layerLayout.setHorizontalSpacing(0)

        controlLayout = QG.QGridLayout()
        controlLabel = HeaderLabel('Control')
        controlLabel.setSizePolicy(QW.QSizePolicy.MinimumExpanding,QW.QSizePolicy.Preferred)
        controlLabel.setMinimumWidth(self.controlWidth)
        controlLayout.addWidget(controlLabel,0,0)
        controlLayout.addWidget(self.toggleControl,1,0)

        displayLayout = QG.QGridLayout()
        displayLayout.addWidget(HeaderLabel(),0,0)
        displayLayout.addWidget(self.toggleDisplay,1,0)
        displayLayout.addWidget(self.controlDisplay,2,0)

        self.layout = QG.QGridLayout(self)
        self.layout.addLayout(layerLayout,0,0)
        self.layout.addLayout(displayLayout,0,1)
        self.layout.addLayout(controlLayout,0,2)

        self.io_buttons.importClicked.connect(self.importImage)
        self.io_buttons.exportClicked.connect(lambda: self.io_buttons.exportData.emit(self.image()))

        # linking stackedDisplay to stackedControl so that items are selected/removed simultaneously.
        self.stackedControl.currentChanged.connect(lambda idx: self.stackedDisplay.setCurrentIndex(idx))
        self.stackedControl.widgetRemoved.connect(lambda idx:
            self.stackedDisplay.removeWidget(self.stackedDisplay.widget(idx)))

        # GSAImage.select controls toggling show / don't show stacked widgets
        self.stackedControl.currentChanged.connect(self.select)

    # Trigger function to allow called import file from main window
    def importTrigger(self):
        return self.io_buttons.importFile()

    # Trigger function to allow called export file from main window
    def exportTrigger(self):
        self.io_buttons.requestExportData()

    # opens file specified by IO importClicked signal
    @errorCheck()
    def importImage(self,filepath):
        self.clear()
        try:
            img = np.array(Image.open(filepath).convert('L'))
        except:
            raise IOError("Cannot read file %s"%filepath)
        if isinstance(img,np.ndarray):
            mod = InitialImage(config=self.config,image=img,width=self.controlWidth)
            self.setWindowTitle(os.path.basename(os.path.basename(filepath)))
            self.addMod(mod)

    # returns image for modification at index. if index is none, returns last image in list.
    @errorCheck()
    def image(self,index=None):
        count = self.stackedControl.count()
        if count > 0:
            if index is None:
                index = count-1
            return self.stackedControl[index].image()
        else:
            raise IOError("No image to return.")

    @errorCheck()
    def clear(self):
        self.stackedControl.clear()
        self.defaultDisplay.clear()

    def removeMod(self):
        if self.stackedControl.count()>0:
            self.stackedControl.removeIndex(self.stackedControl.count()-1)
        if self.stackedControl.count()>0:
            self.stackedControl.setCurrentIndex(self.stackedControl.count()-1)

    @errorCheck(error_text='Error adding layer!')
    def addMod(self,mod=None):
        if mod is None:
            if self.stackedControl.count() > 0:
                mod = self.mod_dict[self.modComboBox.value()](
                    config=self.config,
                    inputMod=self.stackedControl[self.stackedControl.count()-1],
                    width=self.controlWidth)
            else:
                raise ValueError("You need to import an image before adding layers.")
        # Adds modification to stackedControl and display to stackedDisplay
        self.stackedDisplay.addWidget(mod.display())
        self.stackedControl.addWidget(
            mod,
            name=mod.__name__,
            icon=mod.icon(self.mod_list.iconSize()))

        index = self.stackedControl.count()-1
        self.stackedControl.setCurrentIndex(index)

        mod.imageChanged.connect(lambda image: self.stackedControl.setIcon(index,mod.icon(self.mod_list.iconSize())))
        mod.emitImage()

    def select(self, index=None):
        if index==None:
            index = self.stackedControl.currentIndex()
        if index == self.stackedControl.count()-1 and index>=0:
            self.toggleDisplay.setCurrentIndex(1)
            self.toggleControl.setCurrentIndex(1)
            self.controlDisplay.setImageWidget(self.stackedDisplay.widget(index))
        else:
            self.toggleDisplay.setCurrentIndex(0)
            self.toggleControl.setCurrentIndex(0)
            self.controlDisplay.setImageWidget(self.defaultDisplay)
        if index >= 0:
            self.defaultDisplay.setImage(image=self.stackedControl[index].image(),levels=(0,255))
            self.stackedControl[index].update_view()

    def run(self):
        self.show()

class Modification(QW.QScrollArea):
    """
    Abstract class for defining modifications to an image. Modifications form a linked list with each object
    inheriting an input Modification. In this way, images are modified with sequential changes. 


    config:             (ConfigParams) The config parameters for GSA widgets
    inputMod:           (Modification) The Modification that the current object inherits.
    width:              (int) Minimum widget width.

    Signals:
    imageChanged:       (np.ndarray) Signal sent when modification image changes. Returns new image.
    displayChanged:     (NewDisplay, OldDisplay) Signal sent when display is changed. Returns new display and old display.
    """

    imageChanged = QC.pyqtSignal(object) # new image
    displayChanged = QC.pyqtSignal(object,object) # new display, old display
    __name__ = 'Modification'
    def __init__(self,config,inputMod=None,width=None,parent=None):
        super(Modification,self).__init__(parent=parent)
        self.config = config
        if isinstance(width,int):
            self.setMinimumWidth(width)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        self.inputMod = inputMod

        if isinstance(self.inputMod,Modification):
            self.img_out = self.inputMod.image()
        else:
            self.img_out = None

        self._display = ImageWidget(self.image(copy=False))
        self.imageChanged.connect(self._display.setImage)

    def display(self):
        """
        Returns the display widget (either a single ImageWidget or GStackedWidget of ImageWidgets). 
        The display is what is added to GSAImage.stackedDisplay (the main big image area). It is 
        updated by the imageChanged signal unless reimplemented for another purpose.
        """
        return self._display

    def setDisplay(self,display,connectSignal=True):
        oldDisplay = self._display
        if display is not self._display:
            self._display = display

            if connectSignal:
                if hasattr(self._display,'setImage') and callable(self._display.setImage):
                    self.imageChanged.connect(self._display.setImage)
                else:
                    raise ValueError("Display must have method 'setImage' in order to connect imageChanged signal.")
            self.displayChanged.emit(self._display,oldDisplay)

            return oldDisplay
        else:
            return oldDisplay

    def emitImage(self):
        self.imageChanged.emit(self.image())

    def icon(self,qsize):
        """
        Returns a QIcon thumbnail of size dictated by QSize.
        """
        if self.img_out is not None:
            pic = Image.fromarray(self.image(copy=False))
            pic.thumbnail((qsize.width(),qsize.height()))

            data = pic.convert("RGB").tobytes("raw","RGB")

            img = QG.QImage(data, pic.size[0], pic.size[1], QG.QImage.Format_RGB888)
            pix = QG.QPixmap.fromImage(img)
            icon = QG.QIcon(pix)
            icon.addPixmap(pix,QG.QIcon.Selected)

            return icon
        return QG.QIcon()

    def image(self,startImage=False,copy=True):
        """
        Returns the output image after modifications are applied. Or, if startImage==True, return starting image.
        """
        if startImage:
            if self.inputMod is not None:
                return self.inputMod.image(startImage=startImage,copy=copy)
            else:
                return self.image(copy=copy)
        else:
            if self.img_out is None:
                return None
            if copy:
                return self.img_out.copy()
            else:
                return self.img_out

    def setImage(self,img):
        """
        Sets the output image manually. Only necessary for initializing.
        """
        self.img_out = img.astype(np.uint8)
        self.imageChanged.emit(self.img_out)
    def update_image(self):
        """
        (Optional) abstract function for defining and applying modifications to the input image. This is
        used for running the calculations, etc. (not display functionality). It is useful to separate
        the image updating from viewing in many cases which is why it is separate.
        """
        pass
    def update_view(self):
        """
        Updates the image display(s) and other widgets. Emits the update signal(s). Must be implemented
        in any subclass.
        """
        self.update_image()
        self.imageChanged.emit(self.img_out)

class InitialImage(Modification):
    __name__ = 'Initial Image'
    def __init__(self,image,*args,**kwargs):
        super(InitialImage,self).__init__(*args,**kwargs)
        if isinstance(image,np.ndarray):
            self.img_out = image
            self.imageChanged.emit(self.img_out)

class RemoveScale(Modification):
    __name__ = 'Remove Scale Bar'
    def __init__(self,*args,**kwargs):
        super(RemoveScale,self).__init__(*args,**kwargs)
        self.box = None

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
        if crop_img:
            self.img_out = np.array(crop_img)
        else:
            self.img_out = self.inputMod.image()

class ColorMask(Modification):
    __name__ = 'Intensity Mask'
    def __init__(self,*args,**kwargs):
        super(ColorMask,self).__init__(*args,**kwargs)
        self.img_mask = None
        self.img_hist = self.display().imageItem().getHistogram()

        self.histPlot = None
        self.lrItem = None

        self.histPlot = pg.PlotWidget()
        self.histPlot.setMenuEnabled(False)
        self.histPlot.plot(*self.img_hist,
            pen=pg.mkPen(color='k',width=5),
            fillLevel=0,
            brush=(0,0,255,150))
        self.histPlot.getAxis('bottom').setTicks([[(0,'0'),(255,'255')],[]])
        self.histPlot.setXRange(0,255)
        self.histPlot.setLabel(axis='bottom',text='Pixel Intensity')
        self.histPlot.hideAxis('left')

        self.lrItem = pg.LinearRegionItem((0,255),bounds=(0,255))
        self.lrItem.sigRegionChanged.connect(self.update_view)
        self.lrItem.sigRegionChangeFinished.connect(self.update_view)

        self.histPlot.addItem(self.lrItem)
        self.histPlot.setMouseEnabled(False,False)
        self.histPlot.setMaximumHeight(100)

        layout = QG.QGridLayout(self)
        layout.addWidget(self.histPlot,1,0,QC.Qt.AlignTop)

        help = QW.QLabel("Drag the edges of the blue region to set pixel intensity bounds.")
        help.setWordWrap(True)
        layout.addWidget(help,2,0)

        layout.setAlignment(QC.Qt.AlignTop)

    def update_image(self):
        minVal, maxVal = self.lrItem.getRegion()
        img = self.inputMod.image()
        self.img_mask = np.zeros_like(img)
        self.img_mask[np.logical_and(img>minVal,img<maxVal)] = 1
        self.img_out = img*self.img_mask+(1-self.img_mask)*255

    def name(self):
        return 'Color Mask'

class CannyEdgeDetection(Modification):
    __name__ = "Canny Edge Detection"
    def __init__(self,*args,**kwargs):
        super(CannyEdgeDetection,self).__init__(*args,**kwargs)

        self.low_thresh = int(max(self.inputMod.image().flatten())*.1)
        self.high_thresh = int(max(self.inputMod.image().flatten())*.4)
        self.gauss_size = 5

        self.gaussEdit = QG.QLineEdit(str(self.gauss_size))
        self.gaussEdit.setValidator(QG.QIntValidator(3,51))
        self.gaussEdit.setFixedWidth(60)

        self.lowSlider = QG.QSlider(QC.Qt.Horizontal)
        self.lowSlider.setMinimum(0)
        self.lowSlider.setMaximum(255)
        self.lowSlider.setSliderPosition(int(self.low_thresh))
        self.lowEdit = QG.QLineEdit(str(self.low_thresh))
        self.lowEdit.setFixedWidth(60)
        self.lowEdit.setValidator(QG.QIntValidator(0,255))

        self.highSlider = QG.QSlider(QC.Qt.Horizontal)
        self.highSlider.setMinimum(0)
        self.highSlider.setMaximum(255)
        self.highSlider.setSliderPosition(int(self.high_thresh))
        self.highEdit = QG.QLineEdit(str(self.high_thresh))
        self.highEdit.setFixedWidth(60)
        self.highEdit.setValidator(QG.QIntValidator(0,255))

        self.gaussEdit.returnPressed.connect(self._update_sliders)
        self.lowSlider.sliderReleased.connect(self._update_texts)
        self.lowSlider.sliderMoved.connect(self._update_texts)
        self.lowEdit.returnPressed.connect(self._update_sliders)
        self.highSlider.sliderReleased.connect(self._update_texts)
        self.highSlider.sliderMoved.connect(self._update_texts)
        self.highEdit.returnPressed.connect(self._update_sliders)

        layout = QG.QGridLayout(self)
        layout.addWidget(QG.QLabel('Gaussian Size'),0,0)
        layout.addWidget(QG.QLabel('Low Threshold'),1,0)
        layout.addWidget(QG.QLabel('High Threshold'),3,0)
        layout.addWidget(self.gaussEdit,0,1)
        layout.addWidget(self.lowEdit,1,1)
        layout.addWidget(self.highEdit,3,1)
        layout.addWidget(self.lowSlider,2,0,1,2)
        layout.addWidget(self.highSlider,4,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)

    def _update_sliders(self):
        self.gauss_size = int('0'+self.gaussEdit.text())
        self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
        self.gaussEdit.setText(str(self.gauss_size))
        self.low_thresh = int('0'+self.lowEdit.text())
        self.high_thresh = int('0'+self.highEdit.text())

        self.lowSlider.setSliderPosition(self.low_thresh)
        self.highSlider.setSliderPosition(self.high_thresh)

        self.update_view()

    def _update_texts(self):
        self.low_thresh = int(self.lowSlider.value())
        self.high_thresh = int(self.highSlider.value())

        self.lowEdit.setText(str(self.low_thresh))
        self.highEdit.setText(str(self.high_thresh))

        self.update_view()

    def update_image(self):
        self.img_out = cv2.GaussianBlur(self.inputMod.image(),(self.gauss_size,self.gauss_size),0)
        self.img_out = 255-cv2.Canny(self.img_out,self.low_thresh,self.high_thresh,L2gradient=True)

class Dilation(Modification):
    __name__ = "Dilation"
    def __init__(self,*args,**kwargs):
        super(Dilation,self).__init__(*args,**kwargs)
        self.size = 1
        self.sizeEdit = QG.QLineEdit(str(self.size))
        self.sizeEdit.setValidator(QG.QIntValidator(1,20))
        self.sizeEdit.setFixedWidth(60)
        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(1)
        self.sizeSlider.setMaximum(20)
        self.sizeSlider.setSliderPosition(int(self.size))

        self.sizeSlider.valueChanged.connect(self._update_texts)
        self.sizeEdit.returnPressed.connect(self._update_sliders)

        layout = QG.QGridLayout(self)
        layout.addWidget(QG.QLabel('Kernel Size'),0,0)
        layout.addWidget(self.sizeEdit,0,1)
        layout.addWidget(self.sizeSlider,1,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)

    def _update_sliders(self):
        self.size = int('0'+self.sizeEdit.text())
        self.sizeSlider.setSliderPosition(self.size)
        self.update_view()

    def _update_texts(self):
        self.size = int(self.sizeSlider.value())
        self.sizeEdit.setText(str(self.size))
        self.update_view()

    def update_image(self):
        self.img_out = cv2.erode(self.inputMod.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

class Erosion(Modification):
    __name__ = "Erosion"
    def __init__(self,*args,**kwargs):
        super(Erosion,self).__init__(*args,**kwargs)
        self.size = 1
        self.sizeEdit = QG.QLineEdit(str(self.size))
        self.sizeEdit.setValidator(QG.QIntValidator(1,20))
        self.sizeEdit.setFixedWidth(60)
        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(1)
        self.sizeSlider.setMaximum(20)
        self.sizeSlider.setSliderPosition(int(self.size))

        self.sizeSlider.valueChanged.connect(self._update_texts)
        self.sizeEdit.returnPressed.connect(self._update_sliders)

        layout = QG.QGridLayout(self)
        layout.addWidget(QG.QLabel('Kernel Size'),0,0)
        layout.addWidget(self.sizeEdit,0,1)
        layout.addWidget(self.sizeSlider,1,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)

    def _update_sliders(self):
        self.size = int('0'+self.sizeEdit.text())
        self.sizeSlider.setSliderPosition(self.size)
        self.update_view()

    def _update_texts(self):
        self.size = int(self.sizeSlider.value())
        self.sizeEdit.setText(str(self.size))
        self.update_view()

    def update_image(self):
        self.img_out = cv2.dilate(self.inputMod.image(),np.ones((self.size,self.size),np.uint8),iterations=1)

class BinaryMask(Modification):
    __name__ = "Binary Mask"
    def __init__(self,*args,**kwargs):
        super(BinaryMask,self).__init__(*args,**kwargs)

    def update_image(self):
        self.img_out = self.inputMod.image()
        self.img_out[self.img_out < 255] = 0

class Blur(Modification):
    __name__ = "Blur"
    def __init__(self,*args,**kwargs):
        super(Blur,self).__init__(*args,**kwargs)
        self.gauss_size = 5
        self.gaussEdit = QG.QLineEdit(str(self.gauss_size))
        self.gaussEdit.setFixedWidth(100)
        self.gaussEdit.setValidator(QG.QIntValidator(3,51))

        layout = QG.QGridLayout(self)
        layout.addWidget(QG.QLabel('Gaussian Size:'),0,0)
        layout.addWidget(self.gaussEdit,0,1)
        layout.setAlignment(QC.Qt.AlignTop)

        self.gaussEdit.returnPressed.connect(self.update_view)
        self.update_view()

    def update_image(self):
        self.gauss_size = int('0'+self.gaussEdit.text())
        self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
        self.gaussEdit.setText(str(self.gauss_size))
        self.img_out = cv2.GaussianBlur(self.inputMod.image(),(self.gauss_size,self.gauss_size),0)

class MaskingModification(Modification):
    __name__ = "Filter Modification"
    maskChanged = QC.pyqtSignal(object)
    def __init__(self,*args,maskLogic='or',**kwargs):
        Modification.__init__(self,*args,**kwargs)
        assert self.inputMod is not None and isinstance(self.inputMod.image(copy=False),np.ndarray)
        self._mask = np.zeros_like(self.inputMod.image(copy=False),dtype=bool)
        self._mask_out = np.zeros_like(self.inputMod.image(copy=False),dtype=bool)

        self.palette=seaborn.husl_palette(20,l=0.4)
        self.palette=[val for pair in zip(self.palette[:int(len(self.palette)/2)], self.palette[int(len(self.palette)/2):][::-1]) for val in pair]

        self.maskLogic = np.logical_or
        self.updateDisplay(mask=self.mask(copy=False))

        self.maskLogic = np.logical_or if maskLogic.lower() == 'or' else np.logical_and

        self.imageChanged.disconnect()
        self.maskChanged.connect(self.updateDisplay)
        # if isinstance(self.inputMod,MaskingModification):
        #     self.inputMod.maskChanged.connect(lambda _: self.mask(copy=False))

    def emitMask(self):
        self.maskChanged.emit(self.mask(copy=False))

    def image(self,*args,**kwargs):
        try:
            self.img_out = super(MaskingModification,self).image(startImage=True)
            self.img_out[~self.mask(copy=False).astype(bool)] = 255
        except Exception as e:
            # print(e)
            pass

        return super(MaskingModification,self).image(*args,**kwargs)

    def updateDisplay(self,mask=None):
        if mask is None:
            mask = self.mask(copy=False)
        if mask.dtype == bool:
            self.display().setImage(mask_color_img(
                img=self.inputMod.image(startImage=True,copy=False), 
                mask=mask))
        elif mask.dtype == int:
            shaded_img = self.image(startImage=True,copy=True)
            for label in sorted(np.unique(mask)):
                if label == 0:
                    continue
                if label == 1:
                    color = [0,0,255]
                else:
                    color = np.array(self.palette[(label-1)%len(self.palette)])*255
                shaded_img = mask_color_img(
                    img = shaded_img, 
                    mask = mask==label,
                    color = color)
            self.display().setImage(shaded_img)
        # elif mask.dtype == float:
        #     self.display().setImage(mask_color_img(
        #         img=self.inputMod.image(copy=False), 
        #         mask=mask,
        #         color=[255,0,0]))

    def mask(self,copy=True,recursive=False):
        if isinstance(self.inputMod,MaskingModification):
            if recursive:
                mask_in = self.inputMod.mask(copy=False,recursive=True)
            else:
                mask_in = self.inputMod._mask_out
            assert isinstance(mask_in,np.ndarray) and mask_in.shape==self._mask.shape
            mask = self.maskLogic(mask_in,self._mask)
        else:
            mask = self._mask

        if self._mask.dtype==int:
            mask = mask.astype(int)
            idxs = self._mask>0
            mask[idxs] = self._mask[idxs]+1

        self._mask_out = mask
        
        if copy:
            return mask.copy()
        else:
            return mask

class TemplateMatchingWidget(MaskingModification):
    __name__ = "Template Matching"
    def __init__(self,*args,**kwargs):
        super(TemplateMatchingWidget,self).__init__(maskLogic='or',*args,**kwargs)
        self.invert = False

        self.threshSlider = QG.QSlider(QC.Qt.Horizontal)
        self.threshSlider.setMinimum(0)
        self.threshSlider.setMaximum(1000)
        self.threshSlider.setSliderPosition(100)

        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(2)
        self.sizeSlider.setMaximum(30)
        self.sizeSlider.setSliderPosition(15)

        self.invertSelection = QW.QPushButton('Invert Selection')
        self.invertSelection.setIcon(Icon('subtract.svg'))

        self.roi = pg.ROI(
            pos=(0,0),
            size=(20,20),
            removable=True,
            pen=pg.mkPen(color='r',width=3),
            maxBounds=self.display().imageItem().boundingRect(),
            scaleSnap=True,
            snapSize=2)
        self.display().viewBox().addItem(self.roi)

        main_widget = QW.QWidget()
        layout = QG.QGridLayout()
        layout.addWidget(QW.QLabel("Threshold:"),0,0)
        layout.addWidget(self.threshSlider,0,1)
        layout.addWidget(QW.QLabel("Template Size:"),1,0)
        layout.addWidget(self.sizeSlider,1,1)
        layout.addWidget(self.invertSelection,2,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)

        self.invertSelection.clicked.connect(lambda: self.update_view(invert=True))
        self.sizeSlider.valueChanged.connect(lambda v: self.roi.setSize([2*v,2*v]))
        self.roi.sigRegionChangeFinished.connect(lambda: self.update_view())
        self.threshSlider.valueChanged.connect(lambda v: self.update_view(threshold=v))

    def update_image(self,threshold=100,invert=False):
        img_in = self.image(startImage=True,copy=False)
        region = self.roi.getArrayRegion(img_in,self.display().imageItem()).astype(np.uint8)
        x,y = region.shape
        padded_image = cv2.copyMakeBorder(img_in,int(y/2-1),int(y/2),int(x/2-1),int(x/2),cv2.BORDER_REFLECT_101)
        res = cv2.matchTemplate(padded_image,region,cv2.TM_SQDIFF_NORMED)

        threshold = np.logspace(-3,0,1000)[threshold-1]
        
        if invert:
            self._mask = res >= threshold
        else:
            self._mask = res < threshold


    def update_view(self,threshold=None,invert=None):
        if threshold is None:
            threshold=self.threshSlider.value()
        if invert is None:
            invert = self.invert
        else:
            self.invert = ~self.invert
            invert = self.invert
        self.update_image(threshold=threshold,invert=invert)
        self.imageChanged.emit(self.image(copy=False))
        self.maskChanged.emit(self.mask(copy=False))

class CustomFilter(MaskingModification):
    __name__ = "Custom Mask"
    def __init__(self,*args,maskLogic='or',maskVal=True,**kwargs):
        super(CustomFilter,self).__init__(maskLogic=maskLogic,*args,**kwargs)
        self.maskVal = maskVal

        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(2)
        self.sizeSlider.setMaximum(100)
        self.sizeSlider.setSliderPosition(20)
        self.sizeSlider.valueChanged.connect(self.display().imageItem().updateCursor)

        main_widget = QW.QWidget()
        layout = QG.QGridLayout()
        layout.addWidget(QW.QLabel("Cursor Size:"),0,0)
        layout.addWidget(self.sizeSlider,0,1)
        layout.setAlignment(QC.Qt.AlignTop)
        main_widget.setLayout(layout)
        
        self.setWidget(main_widget)

        self.display().imageItem().setDraw(True)
        self.display().imageItem().cursorUpdateSignal.connect(self.update_view)
        self.display().imageItem().dragFinishedSignal.connect(lambda: self.imageChanged.emit(self.image(copy=False)))
        if self.config.mode=='local':
            self.display().viewBox().sigResized.connect(lambda v: self.display().imageItem().updateCursor())
            self.display().viewBox().sigTransformChanged.connect(lambda v: self.display().imageItem().updateCursor())

    def update_view(self,pos=None,scale=None,update_image=False):
        if self.display().imageItem().cursorRadius() is None:
            self.display().imageItem().updateCursor(self.sizeSlider.value())
        if pos is not None and scale is not None:
            shape = self._mask.shape
            ## Cursor position coordinate system is weird so adjustments are made.
            rr, cc = skcircle(shape[0]-pos[1],pos[0],self.sizeSlider.value()*scale,shape=shape)
            self._mask[rr,cc] = self.maskVal

            if update_image == True:
                self.imageChanged.emit(self.image(copy=False))
        else:
            self.imageChanged.emit(self.image(copy=False))
        self.maskChanged.emit(self.mask(copy=False))

class EraseFilter(CustomFilter):
    __name__ = "Erase Mask"
    def __init__(self,*args,**kwargs):
        CustomFilter.__init__(self,maskLogic='and',maskVal=False,*args,**kwargs)
        self._mask = np.ones_like(self.inputMod.image(),dtype=bool)

class ClusterFilter(MaskingModification):
    __name__ = "Abstract Cluster Filter"
    def __init__(self,maskLogic='or',*args,**kwargs):
        MaskingModification.__init__(self,maskLogic=maskLogic,*args,**kwargs)
        self._clusters = None

        self.cluster_list = QW.QListWidget()
        self.cluster_list.setSelectionMode(QG.QAbstractItemView.MultiSelection)
        self.wsize_edit = QW.QLineEdit()
        self.wsize_edit.setValidator(QG.QIntValidator(1,50))
        self.wsize_edit.setText("15")
        self.run_btn = QW.QPushButton("Run")

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
        self._mask = np.zeros_like(self.inputMod.image(copy=False),dtype=int)
        for item in selected_items:
            self._mask[self._clusters==item.data(QC.Qt.UserRole)] = item.data(QC.Qt.UserRole)

    def update_view(self):
        self.update_image()
        self.imageChanged.emit(self.image(copy=False))
        self.maskChanged.emit(self.mask(copy=False))

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


# class UNetFilter(MaskingModification):
#     def __init__(self,*args,**kwargs):
#         super(UNetFilter,self).__init__(*args,**kwargs)

#     def update_image(self):
#         model = load_model(UNET_MODEL_PATH)


class KMeansFilter(ClusterFilter):
    __name__ = "K-Means Clustering"
    def __init__(self,*args,**kwargs):
        ClusterFilter.__init__(self,*args,**kwargs)
        self.n_clusters_edit = QW.QLineEdit()
        self.n_clusters_edit.setValidator(QG.QIntValidator(2,20))
        self.n_clusters_edit.setText("2")

        self.stride_edit = QW.QLineEdit()
        self.stride_edit.setValidator(QG.QIntValidator(1,30))
        self.stride_edit.setText("3")

        self.seed_edit = QW.QLineEdit()
        self.seed_edit.setValidator(QG.QIntValidator(1,1e6))
        self.seed_edit.setText(str(np.random.randint(1e6)))

        main_widget = QW.QWidget()
        layout = QG.QGridLayout(main_widget)
        layout.addWidget(QW.QLabel("Window Size:"),0,0)
        layout.addWidget(self.wsize_edit,0,1)
        layout.addWidget(QW.QLabel("# Clusters:"),1,0)
        layout.addWidget(self.n_clusters_edit,1,1)
        layout.addWidget(QW.QLabel("Stride:"),2,0)
        layout.addWidget(self.stride_edit,2,1)
        layout.addWidget(QW.QLabel("Random Seed:"),3,0)
        layout.addWidget(self.seed_edit,3,1)
        layout.addWidget(self.run_btn,4,0)
        layout.addWidget(BasicLabel('Clusters by Fractional Area:'),5,0,1,2)
        layout.addWidget(self.cluster_list,6,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)
        
        self.setWidget(main_widget)

        self.run_btn.clicked.connect(self.filter)
        self.cluster_list.itemSelectionChanged.connect(self.update_view)

    def filter(self):
        wsize = int('0'+self.wsize_edit.text())
        if wsize % 2 == 0:
            wsize -= 1
        self.wsize_edit.setText(str(wsize))

        stride = int("0"+self.stride_edit.text())

        n_clusters = int('0'+self.n_clusters_edit.text())

        if n_clusters >= 2 and wsize >= 1 and stride >= 1:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=int('0'+self.seed_edit.text()))
            img_in = self.image(startImage=True,copy=False)
            X=util.view_as_windows(
                self.pad(img_in,wsize=wsize,stride=stride),
                window_shape=(wsize,wsize),
                step=stride)
            mask_dim = X.shape[:2]
            X=X.reshape(-1,wsize**2)

            kmeans = kmeans.fit(X)
            mask = kmeans.labels_.reshape(*mask_dim)
            mask = Image.fromarray(mask)
            self._clusters = np.array(mask.resize(img_in.shape[::-1])).astype(np.uint8)+1
            del X

            self.update_list()
            self.update_view()

class GMMFilter(ClusterFilter):
    __name__ = "Gaussian Mixture Clustering"
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
        layout.addWidget(BasicLabel('Clusters by Fractional Area:'),3,0,1,2)
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

        n_components = int('0'+self.n_components_edit.text())

        if n_components >= 2 and wsize >= 1:
            img_in = self.image(startImage=True)

            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                n_init=10
                )
            X = util.view_as_blocks(
                self.pad(img_in,wsize=wsize,stride='block'),
                block_shape=(wsize,wsize)).reshape(-1,wsize**2)
            gmm.fit(X)
            del X
            mod_img=util.view_as_windows(self.pad(img_in,wsize=wsize),window_shape=(wsize,wsize))

            clusters = np.zeros(mod_img.shape[0]*mod_img.shape[1])
            for i in range(mod_img.shape[0]):
                x = mod_img[i,...].reshape(-1,wsize**2)
                
                d = x.shape[0]
                clusters[i*d:(i+1)*d] = gmm.predict(x)

            self._clusters = clusters.reshape(*img_in.shape).astype(np.uint8)+1

            self.update_list()
            self.update_view()

class FilterPattern(Modification):
    __name__ = "Filter Masking"
    def __init__(self,*args,**kwargs):
        super(FilterPattern,self).__init__(*args,**kwargs)
        self.finalMask = {'mask':None}
        self.stackedControl = GStackedWidget(parent=self)

        self.initialImage = InitialImage(config=self.config,image=self.inputMod.image())
        self.initialImage.imageChanged.connect(self.imageChanged.emit)

        self.imageWidgetStack = self.stackedControl.createGStackedWidget()
        self.setDisplay(self.imageWidgetStack,connectSignal=False)

        self._maskClasses = OrderedDict()
        self._maskClasses['Template Match'] = TemplateMatchingWidget
        # self._maskClasses['Gaussian Mixture'] = GMMFilter
        self._maskClasses['K-Means'] = KMeansFilter
        self._maskClasses['Custom'] = CustomFilter
        self._maskClasses['Erase'] = EraseFilter

        self.filterType = QW.QComboBox()
        self.filterType.addItems(list(self._maskClasses.keys()))

        self.filterList = self.stackedControl.createListView()
        self.filterList.setSizePolicy(QW.QSizePolicy.Preferred,QW.QSizePolicy.Maximum)
        self.filterList.setMaximumHeight(100)

        self.addBtn = QG.QPushButton()
        self.addBtn.setIcon(Icon('plus.svg'))
        self.addBtn.setMaximumWidth(48)

        self.removeBtn = QG.QPushButton()
        self.removeBtn.setIcon(Icon('minus.svg'))
        self.removeBtn.setMaximumWidth(48)

        self.exportMask = QG.QPushButton('Export Mask')
        self.exportMask.setIcon(Icon('upload.svg'))

        self.displayImage = ImageWidget(mouseEnabled=(True,True))
        self.displayImage.setMaximumHeight(300)
        self.displayImage.viewBox().autoRange(padding=0)
        self.imageChanged.connect(self.displayImage.setImage)
        self.imageChanged.connect(lambda _: self.displayImage.viewBox().autoRange(padding=0))

        bar_layout = QG.QGridLayout()
        bar_layout.addWidget(self.filterType,0,0)
        bar_layout.addWidget(self.addBtn,0,2)
        bar_layout.addWidget(self.removeBtn,0,1)
        bar_layout.setHorizontalSpacing(0)

        layer_layout = QG.QGridLayout()
        layer_layout.addLayout(bar_layout,0,0)
        layer_layout.addWidget(self.filterList,1,0)
        layer_layout.addWidget(self.exportMask,2,0)
        layer_layout.setAlignment(QC.Qt.AlignTop)
        

        main_layout = QG.QGridLayout(self)
        main_layout.addLayout(layer_layout,0,0)
        main_layout.addWidget(self.stackedControl,1,0)
        main_layout.addWidget(self.displayImage,2,0)
        main_layout.setAlignment(QC.Qt.AlignTop)
        main_layout.setHorizontalSpacing(0)

        self.stackedControl.currentChanged.connect(self.update_view)
        self.addBtn.clicked.connect(self.add)
        self.removeBtn.clicked.connect(self.delete)
        self.exportMask.clicked.connect(self.export)
        self.imageChanged.connect(self.displayImage.setImage)

    def mask(self,copy=True):
        if self.stackedControl.count()>0:
            return self.stackedControl[-1].mask()
        else:
            return np.ones_like(self.image(),dtype=bool)

    def image(self,startImage=False,copy=True):
        if hasattr(self,'stackedControl'): # added this because otherwise initializing the super class messes up.
            if startImage or self.stackedControl.count()==0:
                return Modification.image(self.inputMod,startImage=startImage,copy=copy)
            else:
                return self.stackedControl[-1].image(copy=copy)
        else:
            return Modification.image(self,startImage=startImage,copy=copy)

    def update_view(self,index=None):
        if index is None:
            index = self.stackedControl.currentIndex()

        if index >= 0:
            widget = self.stackedControl[index]
            widget.update_view()
        else:
            self.initialImage.emitImage()

    def add(self):
        method = self.filterType.currentText()
        maskClass = self._maskClasses[method]

        if self.stackedControl.count()>0:
            mod = maskClass(
                config=self.config,
                inputMod=self.stackedControl[-1])
        else:
            mod = maskClass(
                config = self.config,
                inputMod=self.initialImage)

        mod.imageChanged.connect(self.imageChanged.emit)
        mod.maskChanged.connect(lambda _: self.finalMask.update({'mask':self.mask()}))

        self.stackedControl.addWidget(mod,name=method)
        self.imageWidgetStack.addWidget(mod.display())
        self.stackedControl.setCurrentWidget(mod)
        # self.update_view()

    def delete(self):
        if self.stackedControl.count()>0:
            self.stackedControl.removeWidget(self.stackedControl[-1])
            self.emitImage()

    def export(self):
        export_mask = np.zeros_like(self.image(),dtype=np.uint8)
        if self.stackedControl.count()>0:
            export_mask = self.stackedControl.widget(self.stackedControl.count()-1).mask().astype(bool)
            export_mask = export_mask.astype(np.uint8)*255

        default_name = "untitled"
        if self.config.mode == 'local':
            path = os.path.join(os.getcwd(),default_name+"_mask.png")
            name = QW.QFileDialog.getSaveFileName(None, 
                "Export Image", 
                path, 
                "PNG File (*.png)",
                "PNG File (*.png)")[0]
            if name != '' and check_extension(name, [".png"]):
                cv2.imwrite(name,export_mask)
        elif self.config.mode == 'nanohub':
            name = default_name+"_mask.png"
            cv2.imwrite(name,export_mask)
            subprocess.check_output('exportfile %s'%name,shell=True)
        else:
            return

class Crop(Modification):
    __name__ = "Crop"
    def __init__(self,*args,**kwargs):
        super(Crop,self).__init__(*args,**kwargs)

        self.croppedImage = ImageWidget()
        self.croppedImage.setImage(self.image(),levels=(0,255))
        self.displayImage = ImageWidget()
        self.displayImage.setImage(self.image(),levels=(0,255))
        self.setDisplay(self.displayImage,connectSignal=False)

        self.imageChanged.connect(self.croppedImage.setImage)

        self.roi = pg.ROI(
            pos=(0,0),
            size=(32,32),
            removable=True,
            pen=pg.mkPen(color='r',width=2),
            maxBounds=self.displayImage.imageItem().boundingRect())
        self.roi.addScaleHandle(pos=(1,1),center=(0,0))
        self.displayImage.viewBox().addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.update_view)

        layout = QG.QGridLayout(self)
        layout.addWidget(self.croppedImage,0,0)
        layout.setAlignment(QC.Qt.AlignTop)

    def update_image(self):
        img_item = self.displayImage.imageItem()
        self.img_out = self.roi.getArrayRegion(img_item.image,img_item)
        self.img_out = self.img_out.astype(np.uint8)

class DomainCenters(Modification):
    __name__ = "Domain Center Labeling"
    ## This modification is a container for DomainCentersMask so that DomainCentersMask.image functions properly.
    def __init__(self,*args,**kwargs):
        super(DomainCenters,self).__init__(*args,**kwargs)
        self.initialImage = InitialImage(config=self.config,image=self.inputMod.image())
        self.widget = DomainCentersMask(config=self.config,inputMod=self.initialImage)
        self.setDisplay(self.widget.display(),connectSignal=False)

        layout = QG.QGridLayout(self)
        layout.addWidget(self.widget)

class DomainCentersMask(MaskingModification):
    __name__ = "Domain Center Labeling"
    def __init__(self,*args,**kwargs):
        super(DomainCentersMask,self).__init__(*args,**kwargs)
        self._data = {}
        self._data['Domain Center Coordinates'] = []

        self.model = QG.QStandardItemModel()

        self.domain_list = QW.QListView()
        self.domain_list.setModel(self.model)
        self.domain_list.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.domain_list.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        self.domain_list.setMovement(QtWidgets.QListView.Static)
        self.domain_list.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.domain_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)

        self.deleteBtn = QW.QPushButton()
        self.deleteBtn.setIcon(Icon("minus.svg"))
        self.exportBtn = QW.QPushButton("Export")
        self.exportBtn.setIcon(Icon("download.svg"))

        main_widget = QW.QWidget()

        layer_layout = QG.QGridLayout(main_widget)
        layer_layout.addWidget(QW.QLabel("Domain Center Coordinates:"),0,0)
        layer_layout.addWidget(self.domain_list,1,0)
        layer_layout.addWidget(self.deleteBtn,2,0)
        layer_layout.addWidget(self.exportBtn,3,0)
        layer_layout.setAlignment(QC.Qt.AlignTop)

        self.setWidget(main_widget)


        self.domain_list.selectionModel().currentRowChanged.connect(
            lambda current,previous: self.update_view(index=current.row()) if current.isValid() else None)
        self.deleteBtn.clicked.connect(lambda: self.deleteDomain(self.domain_list.selectionModel().currentIndex().row()))
        self.exportBtn.clicked.connect(self.export)
        self.display().imageItem().setDraw(True)
        self.display().imageItem().setEnableDrag(False)
        self.display().imageItem().cursorUpdateSignal.connect(self.update_view)

    def export(self):
        default_name = "untitled.json"
        if self.config.mode == 'local':
            name = QW.QFileDialog.getSaveFileName(None, 
                "Export", 
                default_name, 
                "JSON File (*.json)",
                "JSON File (*.json)")[0]
            if name != '' and check_extension(name, [".json"]):
                with open(name,'w') as f:
                    json.dump(self._data,f)
        elif self.config.mode == 'nanohub':
            path = os.path.join(os.getcwd(),default_name)
            with open(path,'w') as f:
                json.dump(self._data,f)
            print(path)
            subprocess.check_output('exportfile %s'%(path),shell=True)
        else:
            return

    def deleteDomain(self,index=None):
        if index is not None and index>=0:
            self.model.takeRow(index)
            self.update_view()

    def update_view(self,pos=None,scale=None,index=None):
        shape = self.image(copy=False).shape
        if pos is not None:
            x,y = shape[0]-pos[1],pos[0]
            row = self.model.rowCount()
            item = QG.QStandardItem("(%d,%d)"%(x,y))
            self.model.setItem(row,0,item)
            item = QG.QStandardItem()
            item.setData([x,y],QC.Qt.UserRole)
            self.model.setItem(row,1,item)

        self._mask = np.zeros_like(self.image(copy=False),dtype=int)
        for p in range(self.model.rowCount()):
            x,y = self.model.item(p,1).data(QC.Qt.UserRole)
            rr, cc = skcircle(x,y,10,shape=shape)
            if index==p:
                self._mask[rr,cc] = 2
            else:
                self._mask[rr,cc] = 1

        self.updateDisplay(self._mask)

class DrawScale(Modification):
    __name__ = "Draw Scale Bar"
    ## This modification is a container for DrawScaleMask so that DrawScale.image functions properly.
    def __init__(self,*args,**kwargs):
        super(DrawScale,self).__init__(*args,**kwargs)
        self.initialImage = InitialImage(config=self.config,image=self.inputMod.image())
        self.widget = DrawScaleMask(config=self.config,inputMod=self.initialImage)
        self.setDisplay(self.widget.display(),connectSignal=False)

        layout = QG.QGridLayout(self)
        layout.addWidget(self.widget)

    def px_per_um(self):
        return int(1/self.widget.umPerPx)

class DrawScaleMask(MaskingModification):
    def __init__(self,*args,**kwargs):
        super(DrawScaleMask,self).__init__(*args,**kwargs)
        self.newLine = True
        self.startPos = None
        self.endPos = None

        self.drawLengthPx = None
        self.scaleLength = None
        self.umPerPx = None

        self.conversions = OrderedDict()
        self.conversions['mm'] = 0.001
        self.conversions['um'] = 1
        self.conversions['nm'] = 1000

        self.unitsBox = QW.QComboBox()
        self.unitsBox.addItems(list(self.conversions.keys()))
        self.unitsBox.setCurrentIndex(1)
        self.unitsBox.setFixedWidth(60)

        self.pixelLabel = QW.QLabel()
        self.pixelTextLabel = QW.QLabel('Length (px):')
        
        self.scaleLabel = QW.QLabel()
        self.scaleTextLabel = QW.QLabel('Conversion (um/px):')

        self.realLengthEdit = QW.QLineEdit('1')
        self.validator = QG.QDoubleValidator(1e-3,99999,3)
        self.realLengthEdit.setValidator(self.validator)
        self.realLengthLabel = QW.QLabel('Real Length:')

        self.clearBtn = QW.QPushButton("Clear")

        layout = QG.QGridLayout(self)
        layout.addWidget(self.realLengthLabel,0,0)
        layout.addWidget(self.realLengthEdit,0,1)
        layout.addWidget(self.unitsBox,0,2)

        layout.addWidget(self.pixelTextLabel,1,0)
        layout.addWidget(self.pixelLabel,1,2)

        layout.addWidget(self.scaleTextLabel,2,0)
        layout.addWidget(self.scaleLabel,2,2)
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(self.clearBtn,3,0,1,3,QC.Qt.AlignBottom)

        self.emitImage()
        self.display().imageItem().setDraw(True)
        self.display().imageItem().cursorUpdateSignal.connect(self.update_view)
        self.display().imageItem().dragFinishedSignal.connect(self.dragFinished)
        self.realLengthEdit.textChanged.connect(lambda ___: self.updateLabels())
        self.unitsBox.currentIndexChanged.connect(lambda _: self.updateLabels())
        self.clearBtn.clicked.connect(self.clear)

    def dragFinished(self):
        self.newLine = True

    def clear(self):
        self._mask = np.zeros_like(self.image(copy=False),dtype=bool)
        self.newLine = True
        self.update_view()

    def updateLabels(self):
        self.scaleLength = float('0'+self.realLengthEdit.text())/self.conversions[self.unitsBox.currentText()]
        if self.scaleLength is not None and self.drawLengthPx is not None and self.scaleLength > 0 and self.drawLengthPx > 0:
            self.umPerPx = round(self.scaleLength/self.drawLengthPx,5)
            self.scaleLabel.setText(str(self.umPerPx))
            self.pixelLabel.setText(str(self.drawLengthPx))

    def update_view(self,pos=None,scale=None):
        if pos is not None:
            shape = self.image(copy=False).shape
            x,y = shape[0]-pos[1],pos[0]
            x = max(min(shape[0]-1,x),0)
            y = max(min(shape[1]-1,y),0)
            pos = x,y

            if self.newLine:
                self.startPos = pos
                self.newLine = False
            self.endPos = pos

            rr,cc = line(*(self.startPos+self.endPos))

            self._mask = np.zeros_like(self.image(copy=False))
            self._mask[rr,cc] = True
            self._mask = cv2.dilate(self._mask,np.ones((15,15),np.uint8),iterations = 1).astype(bool)

            self.drawLengthPx = int(la.norm(np.array(self.startPos)-np.array(self.endPos)))

        self.updateLabels()
        self.maskChanged.emit(self.mask(copy=False))

class Erase(EraseFilter):
    __name__ = "Erase"
    def __init__(self,*args,**kwargs):
        EraseFilter.__init__(self,*args,**kwargs)

        self.maskChanged.disconnect()
        self.imageChanged.connect(self.display().setImage)

    def updateDisplay(self,mask=None):
        if mask is None:
            mask = self.mask(copy=False)
        if mask.dtype == bool:
            self.display().setImage(mask_color_img(
                img=self.inputMod.image(), 
                mask=mask))

    def update_view(self,pos=None,scale=None,update_image=True):
        EraseFilter.update_view(self,pos,scale,update_image)

    def image(self,*args,**kwargs):
        try:
            self.img_out = super(MaskingModification,self).image(startImage=False)
            self.img_out[~self.mask(copy=False).astype(bool)] = 255
        except Exception as e:
            # print(e)
            pass

        return super(MaskingModification,self).image(*args,**kwargs)

class AlignmentPlots(QW.QWidget):
    def __init__(self,*args,**kwargs):
        super(AlignmentPlots,self).__init__(*args,**kwargs)
        self.data_list = []
        self.images = []
        self.colors = []

        self.sobelSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sobelSizeSlider.setMinimum(1)
        self.sobelSizeSlider.setMaximum(3)
        self.sobelSizeSlider.setSliderPosition(2)

        self.histPlot = pg.PlotWidget(title='Angle Histogram',enableMenu=False,antialias=True)
        # self.histPlot.setMouseEnabled(False,False)
        # self.histPlot.setAspectLocked(True)
        self.histPlot.setXRange(0,180)
        self.histPlot.hideAxis('left')
        self.histPlot.setLabel('bottom',text="Angle (deg)")
        self.histPlot.getAxis('bottom').setTicks([[(tk,str(tk)) for tk in np.linspace(0,180,7).astype(int)],[]])

        self.convPlot = pg.PlotWidget(title='Convolution with Comb Function',enableMenu=False,antialias=True)
        # self.convPlot.setMouseEnabled(False,False)
        # self.convPlot.setAspectLocked(True)
        self.convPlot.setXRange(0,60)
        self.convPlot.hideAxis('left')
        self.convPlot.setLabel('bottom',text="Angle (deg)")
        self.convPlot.getAxis('bottom').setTicks([[(tk,str(tk)) for tk in np.linspace(0,60,5).astype(int)],[]])

        self.tabs = QW.QTabWidget()
        self.tabs.addTab(self.histPlot,'Orientation Histogram')
        self.tabs.addTab(self.convPlot,'Convolution')

        layout = QG.QGridLayout(self)
        layout.addWidget(QW.QLabel("Sobel Size:"),0,0)
        layout.addWidget(self.sobelSizeSlider,0,1)
        layout.addWidget(self.tabs,1,0,1,2)
        layout.setAlignment(QC.Qt.AlignTop)

        self.sobelSizeSlider.valueChanged.connect(lambda _: self.update_view())

    def runSobel(self,images):
        data_list = []
        for image in images:
            sobel_size = 2*int(self.sobelSizeSlider.value())+1
            dx = cv2.Sobel(image,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=sobel_size)
            dy = cv2.Sobel(image,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=sobel_size)

            theta = np.arctan2(dy,dx)*180/np.pi
            magnitude = np.sqrt(dx**2+dy**2)

            values, bin_edges = np.histogram(
                theta.flatten(),
                weights=magnitude.flatten(),
                bins=np.linspace(0,180,181),
                density=True)

            comb = np.zeros(120)
            comb[0] = 1
            comb[60] = 1
            comb[-1] = 1
            convolution = signal.convolve(values,comb,mode='valid')
            convolution = convolution/sum(convolution)
            
            cos = np.average(np.cos(np.arange(len(convolution))*2*np.pi/60),weights=convolution)
            sin = np.average(np.sin(np.arange(len(convolution))*2*np.pi/60),weights=convolution)
            periodic_mean = np.round((np.arctan2(-sin,-cos)+np.pi)*60/2/np.pi).astype(int)
            
            convolution = np.roll(convolution,30-periodic_mean)
            periodic_var = np.average((np.arange(len(convolution))-30)**2,weights=convolution)

            data['Sobel Operator Output'] = {
                "Sobel Operator Size": sobel_size, 
                "Gradient Angle Array": theta.tolist(),
                "Gradient Magnitude Array": magnitude.tolist()}
            data['Edge Orientation Histogram'] = {"Bin Edges (deg)": bin_edges.tolist(), "Values": values.tolist()}
            data['Angular Convolution'] = {
                "Bin Edges (deg)": list(range(0,len(convolution)+1)), 
                "Values": convolution.tolist(),
                "Mean Shift": periodic_mean,
                "Variance": periodic_var}

            data_list.append(data)

        return data_list

    def update_view(self,images=None,colors=None):
        if images is not None and colors is not None:
            assert len(images) == len(colors)
            assert all(isinstance(im,np.ndarray) for im in images)
            assert all(im.shape==images[0].shape for im in images)

            self.images = images
            self.colors = colors

        self.data_list = self.runSobel(images=self.images)

        self.convPlot.clear()
        self.histPlot.clear()
        for c,color in enumerate(self.colors):
            color = pg.mkBrush(color)
            color.setAlpha(100)
            data = self._data_list[c]
            kwargs = {'stepMode':True,'fillLevel':0,'brush':color}
            if len(colors) == 1:
                kwargs['pen'] = pg.mkPen(color='k',width=4)

            self.convPlot.plot(
                data['Angular Convolution']["Bin Edges (deg)"],
                data['Angular Convolution']["Values"],
                **kwargs)

            self.histPlot.plot(
                data['Edge Orientation Histogram']["Bin Edges (deg)"],
                data['Edge Orientation Histogram']["Values"],
                **kwargs)

        return self.data_list

class SegmentCustomFilter(CustomFilter):
    def __init__(self,*args,**kwargs):
        super(SegmentCustomFilter,self).__init__(*args,**kwargs)

class AlignmentSegmentation(Modification):
    __name__="Alignment"
    def __init__(self,*args,**kwargs):
        super(AlignmentSegmentation,self).__init__(*args,**kwargs)
        self.palette=seaborn.husl_palette(30,l=0.4)

        self.segments = GStackedWidget()
        self.segListView = self.segments.createListView()
        self.imageWidgetStack = self.stackedControl.createGStackedWidget()
        self.setDisplay(self.imageWidgetStack,connectSignal=False)

        self.addBtn = QPushButton()
        self.addBtn.setIcon(Icon('plus.svg'))
        self.deleteBtn = QPushButton()
        self.deleteBtn.setIcon(Icon('minus.svg'))

        btn_layout = QG.QGridLayout()
        btn_layout.addWidget(self.addBtn,0,0)
        btn_layout.addWidget(self.deleteBtn,0,1)

        self.plots = AlignmentPlots()

        layout = QG.QGridLayout(self)
        layout.addWidget(self.segListView,0,0)
        layout.addLayout(btn_layout,1,0)
        layout.addWidget(self.segments,2,0)
        layout.addWidget(self.plots,3,0)

        self.addSegmentAct.clicked.connect(self.addSegment)
        self.deleteSegmentAct.clicked.connect(self.deleteSegment)

    def image(self,*args,**kwargs):
        return self.inputMod.image(*args,**kwargs)

    def updateDisplay(self):
        shaded_img = self.image(copy=True)
        images = []
        colors = []
        for label in range(len(self.segments)):
            seg = self.segments[label]
            mask = seg.mask(copy=False)

            image = self.image(copy=True)
            image[~mask] = 255
            images.append(image)

            color = np.array(self.palette[(label-1)%len(self.palette)])*255
            colors.append(colors)

            shaded_img = mask_color_img(
                img = shaded_img, 
                mask = mask,
                color = color)
        self.display().setImage(shaded_img)
        self.plots.update_view(images=images,colors=colors)

    def addSegment(self):
        seg = FilterPattern(
            config=self.config,
            inputMod=InitialImage(config=self.config,image=self.image()))

        seg.maskChanged.connect(lambda _: self.updateDisplay())
        self.imageWidgetStack.addWidget(seg.display())
        self.segments.addWidget(seg,name="New Segment")

        self.segments.setCurrentIndex(len(self.segments)-1)

    def deleteSegment(self,index):
        if len(self.segments)>0:
            self.segments.removeIndex(index)

class Alignment(Modification):
    __name__ = "Alignment"
    def __init__(self,*args,**kwargs):
        super(Alignment,self).__init__(*args,**kwargs)
        self._data = {}

        self.sobelSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sobelSizeSlider.setMinimum(1)
        self.sobelSizeSlider.setMaximum(3)
        self.sobelSizeSlider.setSliderPosition(2)

        self.histPlot = pg.PlotWidget(title='Angle Histogram',enableMenu=False,antialias=True)
        # self.histPlot.setMouseEnabled(False,False)
        # self.histPlot.setAspectLocked(True)
        self.histPlot.setXRange(0,180)
        self.histPlot.hideAxis('left')
        self.histPlot.setLabel('bottom',text="Angle (deg)")
        self.histPlot.getAxis('bottom').setTicks([[(tk,str(tk)) for tk in np.linspace(0,180,7).astype(int)],[]])

        self.convPlot = pg.PlotWidget(title='Convolution with Comb Function',enableMenu=False,antialias=True)
        # self.convPlot.setMouseEnabled(False,False)
        # self.convPlot.setAspectLocked(True)
        self.convPlot.setXRange(0,60)
        self.convPlot.hideAxis('left')
        self.convPlot.setLabel('bottom',text="Angle (deg)")
        self.convPlot.getAxis('bottom').setTicks([[(tk,str(tk)) for tk in np.linspace(0,60,5).astype(int)],[]])

        self.wStd = QtGui.QLabel('')
        self.shiftAngle = QtGui.QLabel('')

        exportHist = QG.QAction('Export Histogram',self)
        exportHist.setIcon(Icon('activity.svg'))
        exportHist.triggered.connect(lambda: self.export(self.histPlot.getPlotItem()))

        exportConv = QG.QAction('Export Convolution',self)
        exportConv.setIcon(Icon('bar-chart-2.svg'))
        exportConv.triggered.connect(lambda: self.export(self.convPlot.getPlotItem()))

        exportData = QG.QAction('Export Data',self)
        exportData.setIcon(Icon('archive.svg'))
        exportData.triggered.connect(lambda: self.exportData())

        self.exportBtn = QW.QPushButton("Export")
        self.exportBtn.setIcon(Icon('upload.svg'))
        exportMenu = QW.QMenu(self.exportBtn)
        exportMenu.addAction(exportHist)
        exportMenu.addAction(exportConv)
        exportMenu.addAction(exportData)

        self.exportBtn.setMenu(exportMenu)
        self.exportBtn.setStyleSheet("QPushButton { text-align: left; }")

        self.plotTabs = QW.QTabWidget()
        self.plotTabs.addTab(self.histPlot,'Angle Histogram')
        self.plotTabs.addTab(self.convPlot,'Convolution')

        layout = QG.QGridLayout(self)
        layout.addWidget(QW.QLabel('Size:'),0,0)
        layout.addWidget(self.sobelSizeSlider,0,1)
        layout.addWidget(self.plotTabs,1,0,1,2)
        layout.addWidget(QW.QLabel('Shifted St. Dev.:'),2,0)
        layout.addWidget(self.wStd,2,1)
        layout.addWidget(self.exportBtn,4,0,1,2)

        self.sobelSizeSlider.valueChanged.connect(self.update_view)
        # self.colors.currentIndexChanged.connect(lambda x: self.update_view())

        # self.update_view()

    def data(self):
        return self._data

    def image(self,*args,**kwargs):
        return super(Alignment,self).image(*args,**kwargs)

    @errorCheck(error_text="Error exporting data!")
    def exportData(self):
        path = os.path.join(os.getcwd(),"untitled.json")
        if self.config.mode == 'local':
            filename = QtWidgets.QFileDialog.getSaveFileName(None,
                "Export Data",
                path,
                "JSON (*.json)",
                "JSON (*.json)")[0]
            with open(filename,'w') as f:
                json.dump(self._data,f)
        else:
            with open(path,'w') as f:
                json.dump(self._data,f)
            subprocess.check_output('exportfile %s'%path,shell=True)


    @errorCheck(error_text="Error exporting item!")
    def export(self,item):
        default_name = "untitled"
        exporter = pyqtgraph.exporters.ImageExporter(item)
        exporter.params.param('width').setValue(1920, blockSignal=exporter.widthChanged)
        exporter.params.param('height').setValue(1080, blockSignal=exporter.heightChanged)
        if self.config.mode == 'local':
            path = os.path.join(os.getcwd(),default_name+".png")
            name = QtWidgets.QFileDialog.getSaveFileName(None, 
                "Export Image", 
                path, 
                "PNG File (*.png)",
                "PNG File (*.png)")[0]
            if name != '' and check_extension(name, [".png"]):
                exporter.export(fileName=name)
        elif self.config.mode == 'nanohub':
            path = os.path.join(os.getcwd(),default_name+".png")
            exporter.export(fileName=path)
            subprocess.check_output('exportfile %s'%path,shell=True)
        else:
            return

    def update_image(self):
        sobel_size = 2*int(self.sobelSizeSlider.value())+1
        self.dx = cv2.Sobel(self.inputMod.image(),ddepth=cv2.CV_64F,dx=1,dy=0,ksize=sobel_size)
        self.dy = cv2.Sobel(self.inputMod.image(),ddepth=cv2.CV_64F,dx=0,dy=1,ksize=sobel_size)


        theta = np.arctan2(self.dy,self.dx)*180/np.pi
        magnitude = np.sqrt(self.dx**2+self.dy**2)

        values, bin_edges = np.histogram(
            theta.flatten(),
            weights=magnitude.flatten(),
            bins=np.linspace(0,180,181),
            density=True)

        comb = np.zeros(120)
        comb[0] = 1
        comb[60] = 1
        comb[-1] = 1
        convolution = signal.convolve(values,comb,mode='valid')
        convolution = convolution/sum(convolution)
        
        cos = np.average(np.cos(np.arange(len(convolution))*2*np.pi/60),weights=convolution)
        sin = np.average(np.sin(np.arange(len(convolution))*2*np.pi/60),weights=convolution)
        periodic_mean = np.round((np.arctan2(-sin,-cos)+np.pi)*60/2/np.pi).astype(int)
        
        convolution = np.roll(convolution,30-periodic_mean)
        periodic_var = np.average((np.arange(len(convolution))-30)**2,weights=convolution)

        self._data['Sobel Operator Output'] = {
            "Sobel Operator Size": sobel_size, 
            "Gradient Angle Array": theta.tolist(),
            "Gradient Magnitude Array": magnitude.tolist()}
        self._data['Edge Orientation Histogram'] = {"Bin Edges (deg)": bin_edges.tolist(), "Values": values.tolist()}
        self._data['Angular Convolution'] = {
            "Bin Edges (deg)": list(range(0,len(convolution)+1)), 
            "Values": convolution.tolist(),
            "Mean Shift": int(periodic_mean),
            "Variance": float(periodic_var)}

    def update_view(self):
        self.update_image()
        color = pg.mkColor('b')
        color.setAlpha(150)

        self.convPlot.clear()
        self.convPlot.plot(
            self._data['Angular Convolution']["Bin Edges (deg)"],
            self._data['Angular Convolution']["Values"],
            stepMode=True,
            fillLevel=0,
            brush=color,
            pen=pg.mkPen(color='k',width=4))
        self.convPlot.addLine(x=30)
        self.convPlot.addLine(x=30-np.sqrt(self._data['Angular Convolution']["Variance"]),pen=pg.mkPen(dash=[3,5],width=4))
        self.convPlot.addLine(x=30+np.sqrt(self._data['Angular Convolution']["Variance"]),pen=pg.mkPen(dash=[3,5],width=4))
        self.histPlot.clear()

        self.histPlot.plot(
            self._data['Edge Orientation Histogram']["Bin Edges (deg)"],
            self._data['Edge Orientation Histogram']["Values"],
            stepMode=True,
            fillLevel=0,
            brush=color,
            pen=pg.mkPen(color='k',width=4))
        self.wStd.setNum(np.sqrt(self._data['Angular Convolution']["Variance"]))

        self.imageChanged.emit(np.array(self._data['Sobel Operator Output']["Gradient Magnitude Array"]))
       
def main():
    nargs = len(sys.argv)
    if nargs > 1:
        mode = sys.argv[1]
    else:
        mode = 'local'
    if mode not in ['nanohub','local']:
        mode = 'local'

    REPO_DIR = "."
    if mode == 'local':
        REPO_DIR = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')
    else:
        if os.environ.get("RUN_LOCATION"):
            REPO_DIR = os.environ.get("RUN_LOCATION")
    
    app = QG.QApplication([])
    # img_analyzer = GSAImage(mode=mode)
    # img_analyzer.run()
    main = Main(mode=mode, repo_dir = REPO_DIR)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

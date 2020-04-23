from __future__ import division
import numpy as np
import scipy as sc
import cv2, sys, time, json, copy, subprocess, os
from skimage import transform
from skimage import util
from skimage import color
import functools
from skimage.draw import circle as skcircle
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
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
from util.gwidgets import GStackedWidget, ImageWidget, StandardDisplay,HeaderLabel,SubheaderLabel,MaxSpacer,SmartImageItem,SpacerMaker,ControlImageWidget
from util.io import IO
from util.icons import Icon
from io import BytesIO
import traceback

pg.setConfigOption('background', 'w')
pg.setConfigOption('imageAxisOrder', 'row-major')

QW=QtWidgets
QC=QtCore
QG=QtGui

testMode = True

## TODO: add toolbar and/or menu bar
class Main(QW.QMainWindow):
    """
    Main window containing the GSAImage widget. Adds menu bar / tool bar functionality.
    """
    def __init__(self,mode='local',*args,**kwargs):
        super(Main,self).__init__(*args,**kwargs)

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
        clearAction.triggered.connect(self.mainWidget.clear)

        exitAction = QG.QAction("&Exit",self)
        exitAction.setIcon(Icon('log-out.svg'))
        exitAction.triggered.connect(self.close)
        
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(importAction)
        fileMenu.addAction(exportAction)
        fileMenu.addAction(clearAction)
        fileMenu.addAction(exitAction)

        aboutAction = QG.QAction("&About",self)
        aboutAction.setIcon(Icon('info.svg'))
        aboutAction.triggered.connect(self.showAboutDialog)

        helpMenu = mainMenu.addMenu('&Help')
        helpMenu.addAction(aboutAction)

        self.show()

    def showAboutDialog(self):
        about_dialog = QW.QMessageBox(self)
        about_dialog.setText("About This Tool")
        about_dialog.setWindowModality(QC.Qt.WindowModal)

        # Needs text
        about_text = """
        """

        about_dialog.setInformativeText(about_text)
        about_dialog.exec()


class GSAImage(QW.QWidget):
    def __init__(self,mode='local',parent=None):
        super(GSAImage,self).__init__(parent=parent)
        self.config = ConfigParams(mode=mode)
        self.workingDirectory = os.getcwd()
        self.controlWidth = 250

        if self.config.mode == 'nanohub':
            if 'TempGSA' not in os.listdir(self.workingDirectory):
                os.mkdir('TempGSA')
            self.tempdir = os.path.join(os.getcwd(),'TempGSA')
            os.chdir(self.tempdir)
            self.workingDirectory = os.path.join(self.workingDirectory,'TempGSA')

        # Ordered dictionary links to modifications. Used ordered dict so that
        # controlling combobox order more natural 
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

        # Add / remove modification buttons
        self.addBtn = QG.QPushButton()
        self.addBtn.setIcon(Icon("plus.svg"))
        self.addBtn.setMaximumWidth(32)
        self.addBtn.setMaximumHeight(32)
        self.addBtn.clicked.connect(lambda: self.addMod(mod=None))

        self.removeBtn = QG.QPushButton()
        self.removeBtn.setIcon(Icon("minus.svg"))
        self.removeBtn.setMaximumWidth(32)
        self.removeBtn.setMaximumHeight(32)
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
        layerLayout = QG.QGridLayout()
        layerLayout.addWidget(HeaderLabel('Layers'),0,0)
        layerLayout.addWidget(self.mod_list,2,0,1,3)
        layerLayout.addWidget(self.modComboBox,3,0)
        layerLayout.addWidget(self.addBtn,3,2)
        layerLayout.addWidget(self.removeBtn,3,1)

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
            img = cv2.imread(filepath)
        except:
            raise IOError("Cannot read file %s"%os.path.basename(filepath))
        if isinstance(img,np.ndarray):
            mod = InitialImage(width=self.controlWidth)
            mod.setImage(img)
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
                    width=self.controlWidth,
                    properties={'mode':self.config.mode})
            else:
                raise ValueError("You need to import an image before adding layers.")
        # Adds modification to stackedControl and display to stackedDisplay
        self.stackedDisplay.addWidget(mod.display())
        self.stackedControl.addWidget(
            mod,
            name=mod.name(),
            icon=mod.icon(self.mod_list.iconSize()))

        index = self.stackedControl.count()-1
        self.stackedControl.setCurrentIndex(index)

        mod.imageChanged.connect(lambda image: self.stackedControl.setIcon(index,mod.icon(self.mod_list.iconSize())))
        mod.emitImage()

    def select(self, index):
        if index == self.stackedControl.count()-1 and index>=0:
            self.toggleDisplay.setCurrentIndex(1)
            self.toggleControl.setCurrentIndex(1)
            self.controlDisplay.setImageWidget(self.stackedDisplay.widget(index))
        else:
            self.toggleDisplay.setCurrentIndex(0)
            self.toggleControl.setCurrentIndex(0)
            self.defaultDisplay.setImage(self.stackedControl[index].image(),levels=(0,255))
            self.controlDisplay.setImageWidget(self.defaultDisplay)
        if index >= 0:
            self.stackedControl[index].update_view()

    def widget(self):
        return self

    def run(self):
        self.show()


class Modification(QW.QScrollArea):
    """
    Abstract class for defining modifications to an image. Modifications form a linked list with each object
    inheriting an input Modification. In this way, images are modified with sequential changes. 

    inputMod:           (Modification) the Modification that the current object inherits.
    properties:         (dict) a dictionary of properties that may be used for the modification.

    Signals:
    imageChanged:       (np.ndarray) Signal sent when modification image changes. Returns new image.
    displayChanged:     (NewDisplay, OldDisplay) Signal sent when display is changed. Returns new display and old display.


    Subclassing:
    If a different display is to be used for GSAImage.stackedDisplay, subclasses should use Modification.setDisplay
    to use a different display and typically the imageChanged signal should connect to it in order to update it. 
    """

    imageChanged = QC.pyqtSignal(object)
    displayChanged = QC.pyqtSignal(object,object)
    def __init__(self,inputMod=None,width=None,properties={},parent=None):
        super(Modification,self).__init__(parent=parent)
        if isinstance(width,int):
            self.setMinimumWidth(width)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        self.inputMod = inputMod
        self.properties = properties

        self._display = ImageWidget()
        self.imageChanged.connect(self._display.setImage)

        if inputMod is not None:
            self.setImage(self.inputMod.image())
        else:
            self.img_out = None

    def display(self):
        """
        Returns the display widget (usually an ImageWidget instance). The display is what is 
        added to GSAImage.stackedDisplay (the main big image area). It is updated by the 
        imageChanged signal unless reimplemented for another purpose.
        """
        return self._display

    def setDisplay(self,display,connectSignal=True):
        try:
            pass
            # self.imageChanged.disconnect()
        except:
            pass
        # display.setMaximumWidth(300)
        self._display, oldDisplay = display, self._display

        if connectSignal:
            self.imageChanged.connect(self._display.setImage)
        self.displayChanged.emit(self._display,oldDisplay)

        return oldDisplay

    def emitImage(self):
        self.imageChanged.emit(self.image())

    def icon(self,qsize):
        """
        Returns a QIcon thumbnail of size dictated by QSize.
        """
        if self.img_out is not None:
            pic = Image.fromarray(self.img_out)
            pic.thumbnail((qsize.width(),qsize.height()))

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
    def image(self,copy=True):
        """
        Returns the output image after modifications are applied.
        """
        if copy:
            return self.img_out.copy()
        else:
            return self.img_out
    def name(self):
        return 'Default Modification'
    def setImage(self,img):
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
    def __init__(self,*args,**kwargs):
        super(RemoveScale,self).__init__(*args,**kwargs)

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
    def __init__(self,*args,**kwargs):
        super(ColorMask,self).__init__(*args,**kwargs)
        self.img_mask = None
        self.img_hist = self.display().imageItem().getHistogram()

        self.histPlot = None
        self.lrItem = None

        self.histPlot = pg.PlotWidget()
        self.histPlot.plot(*self.img_hist,
            pen=pg.mkPen(color='k',width=5),
            fillLevel=0,
            brush=(0,0,255,150))
        self.histPlot.getAxis('bottom').setTicks([[(0,'0'),(255,'255')],[]])
        self.histPlot.setXRange(0,255)
        self.histPlot.setLabel(axis='bottom',text='Pixel Intensity')
        self.histPlot.hideAxis('left')

        self.lrItem = pg.LinearRegionItem((0,255),bounds=(0,255),pen=pg.mkPen(color='r',width=5))
        self.lrItem.sigRegionChanged.connect(self.update_view)
        self.lrItem.sigRegionChangeFinished.connect(self.update_view)

        self.histPlot.addItem(self.lrItem)
        self.histPlot.setMouseEnabled(False,False)
        self.histPlot.setMaximumHeight(100)

        layout = QG.QGridLayout(self)
        layout.addWidget(self.histPlot,1,0,QC.Qt.AlignTop)

        help = QW.QLabel("Drag the red edges of the blue region to set pixel intensity bounds.")
        help.setWordWrap(True)
        layout.addWidget(help,2,0)

        layout.setAlignment(QC.Qt.AlignTop)


    def widget(self):
        return self.histPlot

    def update_image(self):
        minVal, maxVal = self.lrItem.getRegion()
        img = self.inputMod.image()
        self.img_mask = np.zeros_like(img)
        self.img_mask[np.logical_and(img>minVal,img<maxVal)] = 1
        self.img_out = img*self.img_mask+(1-self.img_mask)*255

    def name(self):
        return 'Color Mask'

class CannyEdgeDetection(Modification):
    def __init__(self,*args,**kwargs):
        super(CannyEdgeDetection,self).__init__(inputMod,properties)

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

    def name(self):
        return 'Canny Edge Detection'

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

    def widget(self):
        return self

class Dilation(Modification):
    def __init__(self,inputMod,properties={}):
        super(Dilation,self).__init__(inputMod,properties)
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

    def name(self):
        return 'Dilation'

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

    def widget(self):
        return self.control

class Erosion(Modification):
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

    def name(self):
        return 'Erosion'

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

    def widget(self):
        return self

class BinaryMask(Modification):
    def __init__(self,*args,**kwargs):
        super(BinaryMask,self).__init__(*args,**kwargs)

    def name(self):
        return 'Binary Mask'

    def update_image(self):
        self.img_out = self.inputMod.image()
        self.img_out[self.img_out < 255] = 0

class Blur(Modification):
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

    def widget(self):
        return self

    def name(self):
        return 'Blur'

class FilterModification(Modification):
    maskChanged = QC.pyqtSignal(object)
    def __init__(self,sequential=True,*args,**kwargs):
        super(Modification,self).__init__(*args,**kwargs)
        self._mask_in = np.array([])
        self._mask_out = np.array([])
        self.sequential = sequential

        if hasattr(self.inputMod,mask) and callable(self.inputMod.mask):
            self._mask_in = self.inputMod.mask()
        if hasattr(self.inputMod,maskChanged) and isinstance(self.inputMod.maskChanged,QC.pyqtSignal):
            self.inputMod.maskChanged.connect(self.setMask)

        self.workingImage = ImageWidget()
        self.setDisplay(self.workingImage,connectSignal=False)

    def setMask(self,mask):
        self._mask_in = mask

    def mask(self,copy=True):
        if copy:
            return self._mask_out.copy()
        else:
            return self._mask_out

class TemplateMatchingWidget(FilterModification):
    def __init__(self,*args,**kwargs):
        FilterModification.__init__(self,sequential=False,*args,**kwargs)
        self._mask_in = mask_in
        self._mask = np.zeros_like(self.inputMod.image(),dtype=bool)
        self.img_item=img_item
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
        self.img_item = img_item
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
        self.img_item = img_item
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
    def __init__(self,inputMod,properties={}):
        super(FilterPattern,self).__init__(inputMod,properties)

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

        self.filterImage = ImageWidget(smart=True,mouseEnabled=(True,True))
        self.filterViewBox = self.filterImage.viewBox()
        self.filterImageItem = self.filterImage.imageItem()
        print(self.inputMod.image().shape)
        self.filterImage.setImage(self.inputMod.image()[:,::-1],levels=(0,255))
        self.filterViewBox.sigResized.connect(lambda v: self.filterImageItem.updateCursor())
        self.filterViewBox.sigTransformChanged.connect(lambda v: self.filterImageItem.updateCursor())
        self.setDisplay(self.filterImage)

        self.displayImage = ImageWidget(mouseEnabled=(True,True))
        self.imageChanged.connect(lambda image: self.displayImage.setImage(image,levels=(0,255)))
        
        self.blankWidget = QW.QWidget()
        self.stackedControl = QW.QStackedWidget()

        self.toggleControl = QW.QStackedWidget()
        self.toggleControl.addWidget(self.blankWidget)
        self.toggleControl.addWidget(self.stackedControl)

        layer_layout = QG.QGridLayout()
        layer_layout.addWidget(self.wFilterList,0,0,6,1)
        layer_layout.addWidget(self.wFilterType,0,1)
        layer_layout.addWidget(self.wAdd,1,1)
        layer_layout.addWidget(self.wRemove,2,1)
        layer_layout.addWidget(self.wExportMask,4,1)
        layer_layout.setAlignment(QC.Qt.AlignTop)
        
        main_layout = QG.QGridLayout(self)
        main_layout.addLayout(layer_layout,0,0)
        main_layout.addWidget(self.toggleControl,1,0)
        main_layout.addWidget(self.displayImage,2,0)
        main_layout.setAlignment(QC.Qt.AlignTop)

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

            self.displayImage.setImage(widget.image(),levels=(0,255))
        else:
            self.toggleControl.setCurrentWidget(self.blankWidget)
            self.displayImage.setImage(self.image(),levels=(0,255))
            self.displayImage.setImage(self.image(),levels=(0,255))

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
            mask_widget = maskClass(
                inputMod=in_mod,
                img_item=self.filterImageItem,
                vbox=self.filterViewBox,
                mask_in=mask_in,
                img_in=self.inputMod.image())
        else:
            mask_widget = maskClass(
                inputMod=in_mod,
                img_item=self.filterImageItem,
                mask_in=mask_in,
                img_in=self.inputMod.image())

        mask_widget.imageChanged.connect(self.imageChanged.emit)
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
                self.filterViewBox.removeItem(w.roi)
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
    def __init__(self,inputMod,properties={}):
        super(Crop,self).__init__(inputMod,properties)

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
        layout.setAlignment(QC.Qt.AlignTop)
        layout.addWidget(self.croppedImage,0,0)

    def update_image(self):
        img_item = self.displayImage.imageItem()
        self.img_out = self.roi.getArrayRegion(img_item.image,img_item)
        self.img_out = self.img_out.astype(np.uint8)

    def widget(self):
        return self

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
    def __init__(self,*args,**kwargs):
        super(Erase,self).__init__(*args,**kwargs)
        self.img_out = self.inputMod.image()
        self.eraser_size = 10
        

        self.wImgBox = pg.GraphicsLayoutWidget()
        self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
        self.wImgROI = pg.ImageItem()
        self.wImgROI.setImage(self.img_out,levels=(0,255))
        self.wImgBox_VB.addItem(self.wImgROI)
        self.wImgBox_VB.setAspectLocked(True)

        self.sizeSlider = QG.QSlider(QC.Qt.Horizontal)
        self.sizeSlider.setMinimum(1)
        self.sizeSlider.setMaximum(100)
        self.sizeSlider.setSliderPosition(self.eraser_size)

        kern = (np.ones((self.eraser_size,self.eraser_size))*255).astype(np.uint8)
        self.wImgROI.setDrawKernel(kern, mask=None, center=(int(self.eraser_size/2),int(self.eraser_size/2)), mode='set')
        self.sizeSlider.valueChanged.connect(self.update_view)

        self.wLayout = QtGui.QGridLayout(self)
        self.wLayout.addWidget(QG.QLabel('Eraser Size:'),0,0)
        self.wLayout.addWidget(self.sizeSlider,0,1)
        self.wLayout.addWidget(self.wImgBox,1,0,4,4)
        self.wLayout.setAlignment(QtCore.Qt.AlignTop)

    def update_image(self):
        self.eraser_size = int(self.sizeSlider.value())
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

        self.histPlot = pg.PlotWidget(title='Angle Histogram',pen=pg.mkPen(color='k',width=4))
        self.histPlot.setXRange(0,180)
        self.histPlot.hideAxis('left')

        self.wConvPlot = pg.PlotWidget(title='Convolution with Comb Function',pen=pg.mkPen(color='k',width=4))
        self.wConvPlot.setXRange(0,60)
        self.wConvPlot.hideAxis('left')

        self.wStd = QG.QLabel('')

        self.exportHistBtn = QG.QPushButton('Export Histogram')
        self.exportConvBtn = QG.QPushButton('Export Convolution')

        self.wLayout.addWidget(QG.QLabel('Size:'),0,0)
        self.wLayout.addWidget(self.wSobelSizeSlider,0,1)

        self.wLayout.addWidget(self.histPlot,4,0,4,4)
        self.wLayout.addWidget(self.wConvPlot,8,0,4,4)
        self.wLayout.addWidget(QG.QLabel('Shifted St. Dev.:'),12,0)
        self.wLayout.addWidget(self.wStd,12,1)
        self.wLayout.addWidget(self.exportHistBtn,13,0)
        self.wLayout.addWidget(self.exportConvBtn,13,1)

        self.wSobelSizeSlider.valueChanged.connect(self.update_view)

        self.update_view()
        self.exportHistBtn.clicked.connect(lambda: self.export(self.histPlot.getPlotItem()))
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
        self.histPlot.clear()
        self.histPlot.plot(
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

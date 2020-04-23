import os
import numpy as np
from PIL import Image
from PyQt5 import QtGui, QtCore, QtWidgets
import copy
import io
import traceback
import functools
import logging
from collections.abc import Sequence
from collections import OrderedDict, deque
import pyqtgraph as pg
from .icons import Icon

logger = logging.getLogger(__name__)

QC = QtCore
QW = QtWidgets
QG = QtGui

class Label(QtWidgets.QLabel):
    def __init__(self,text='',tooltip=None):
        super(Label,self).__init__()
        self.tooltip = tooltip
        if text is None:
            text = ''
        self.setText(text)
        if isinstance(self.tooltip,str):
            self.setMouseTracking(True)

    def setMouseTracking(self,flag):
        """
        Ensures mouse tracking is allowed for label and all parent widgets 
        so that tooltip can be displayed when mouse hovers over it.
        """
        QtWidgets.QWidget.setMouseTracking(self, flag)
        def recursive(widget,flag):
            try:
                if widget.mouseTracking() != flag:
                    widget.setMouseTracking(flag)
                    recursive(widget.parent(),flag)
            except:
                pass
        recursive(self.parent(),flag)

    def event(self,event):
        if event.type() == QtCore.QEvent.Leave:
            QtWidgets.QToolTip.hideText()
        return QtWidgets.QLabel.event(self,event)

    def mouseMoveEvent(self,event):
        QtWidgets.QToolTip.showText(
            event.globalPos(),
            self.tooltip
            )

class LabelMaker:
    def __init__(self,family=None,size=None,bold=False,italic=False):
        self.family = family
        self.size = size
        self.bold = bold
        self.italic = italic

    def __call__(self,text='',tooltip=None):
        label = Label(text,tooltip=tooltip)
        font = QtGui.QFont()
        if self.family:
            font.setFamily(self.family)
        if self.size:
            font.setPointSize(self.size)
        if self.bold:
            font.setBold(self.bold)
        if self.italic:
            font.setItalic(self.italic)
        label.setFont(font)

        return label


class SpacerMaker:
    def __init__(self,vexpand=True,hexpand=True,width=None,height=None):
        if vexpand == True:
            self.vexpand = QtGui.QSizePolicy.Ignored
        else:
            self.vexpand = QtGui.QSizePolicy.Preferred

        if hexpand == True:
            self.hexpand = QtGui.QSizePolicy.Ignored
        else:
            self.hexpand = QtGui.QSizePolicy.Preferred

        self.width = width
        self.height = height


    def __call__(self):
        if isinstance(self.width,int):
            width = self.width
        else:
            width = BasicLabel().sizeHint().width()
        if isinstance(self.height,int):
            height = self.height
        else:
            height = BasicLabel().sizeHint().height()

        spacer = QtGui.QSpacerItem(
            width,
            height,
            vPolicy=self.vexpand,
            hPolicy=self.hexpand
        )

        return spacer


class ConfirmationBox(QtWidgets.QMessageBox):
    okSignal = QtCore.pyqtSignal()
    cancelSignal = QtCore.pyqtSignal()
    def __init__(self,question_text,informative_text=None,parent=None):
        super(ConfirmationBox,self).__init__(self,parent=parent)
        assert isinstance(question_text,str)

        self.setText(question_text)
        if informative_text:
            self.setInformativeText(informative_text)
        self.setStandardButtons(
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        self.setWindowModality(QtCore.Qt.WindowModal)

        self.buttonClicked.connect(self.onClick)

    def onClick(self,btn):
        if btn.text() == "OK":
            self.okSignal.emit()
        else:
            self.cancelSignal.emit()

class GOrderedDict(OrderedDict):
    def __getitem__(self,key):
        if isinstance(key,int) and key not in self.keys():
            key = list(self.keys())[key]
        return super(GOrderedDict,self).__getitem__(self,key)

class GStackedMeta(type(QtWidgets.QStackedWidget),type(Sequence)):
    pass

class GStackedWidget(QtWidgets.QStackedWidget,Sequence,metaclass=GStackedMeta):
    """
    A much better version of QStackedWidget. Has more functionality including:

        - creates lists that are linked to GStackedWidget
        - is indexable (i.e. w = stackedwidget[i] will get you the i'th widget)
        - can make it autosignal a focus/update function
        - existing functions have more flexibility

    border:         (bool) Whether the widget should have a border.
    """
    def __init__(self,border=False,parent=None):
        QtWidgets.QStackedWidget.__init__(self,parent=parent)
        if border == True:
            self.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.model = QtGui.QStandardItemModel()

    def __getitem__(self,key):
        if isinstance(key,str):
            try:
                key = self.model.index(key)
            except:
                raise KeyError("Key '%s' is not in model."%key)
        if key < self.count():
            return self.widget(key)
        else:
            raise IndexError("Index %s out of range for GStackedWidget with length %s"%(key,self.count()))

    def __len__(self):
        return self.count()

    def setIcon(self,row,icon):
        item = self.model.item(row)
        item.setData(icon,QtCore.Qt.DecorationRole)

    def createListView(self):
        listview = QtWidgets.QListView()
        listview.setModel(self.model)
        listview.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        listview.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        listview.setMovement(QtWidgets.QListView.Static)
        listview.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        listview.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        # listview.setUniformItemSizes(True)

        self.currentChanged.connect(lambda index: listview.setCurrentIndex(self.model.createIndex(index,0)))

        listview.selectionModel().currentRowChanged.connect(
            lambda current,previous: self.setCurrentIndex(current.row()) if current.isValid() else None)

        return listview

    def createGStackedWidget(self,*args,**kwargs):
        gstacked = GStackedWidget(*args,**kwargs)
        self.widgetRemoved.connect(lambda i: gstacked.removeIndex(i) if i < gstacked.count() else None)
        self.currentChanged.connect(lambda i: gstacked.setCurrentIndex(i) if i < gstacked.count() else None)

        return gstacked

    def addWidget(self,widget,name=None,icon=QtGui.QIcon(),focus_slot=None):
        """
        Add a widget to the GStackedWidget. Name and icon are typically used for the listview.

        widget:         (QWidget) Widget to be added.
        name:           (str) Used for labeling widgets.
        icon:           (QIcon) Used for setting a corresponding icon
        focus_slot:     (callable) Callable function to be signaled when widget selected.
        """
        QtWidgets.QStackedWidget.addWidget(self,widget)
        
        if callable(focus_slot):
            self.currentChanged.connect(lambda i: focus_slot() if self[i]==widget else None)
        elif focus_slot is not None:
            raise TypeError("Parameter 'focus_slot' must be a callable function!")

        if not isinstance(name,str):
            name = "%s - %s"%(widget.__class__.__name__,self.count()-1)

        item = QtGui.QStandardItem()
        item.setIcon(icon)
        item.setText(name)
        self.model.appendRow(item)

        return self.count()-1

    def removeIndex(self,index):
        """
        Remove widget by index.

        index:          (int) Index of widget.
        """
        self.model.takeRow(index)
        QtWidgets.QStackedWidget.removeWidget(self,self.widget(index))

    def removeCurrentWidget(self):
        index = self.currentIndex()
        self.removeIndex(index)

    def index(self,key):
        """
        Index of widget name.
        """
        return self.model.findItems(key)[0].row()

    def clear(self):
        while self.count()>0:
            self.removeIndex(0)

    def setCurrentIndex(self,key):
        """
        Set current widget by index or item label.

        key:            (str or int) Key is the index or the item label of the widget.
        """
        if isinstance(key,int):
            pass
        elif isinstance(key,str):
            key = self.index(key)
        else:
            raise ValueError("Parameter 'key' must be of type 'int' or 'str'. Found type '%s'."%type(key))
        QtWidgets.QStackedWidget.setCurrentIndex(self,key)

    def widget(self,key):
        """
        Get widget by index or item label.

        key:            (str or int) Key is the index or the item label of the widget.
        """
        if isinstance(key,int):
            widget = super(GStackedWidget,self).widget(key)
        elif isinstance(key,str):
            try:
                key = self.index(key)
                widget = super(GStackedWidget,self).widget(key)
            except:
                raise ValueError("Key '%s' not in model!"%key)
        else:
            raise ValueError("Key '%s' is type '%s'. Keys must be type 'int' or 'str'!"%(key,type(key)))
        return widget

class ImageWidget(pg.GraphicsLayoutWidget):
    def __init__(self, path=None,smart=True,mouseEnabled=(True,True),aspectLocked=True,*args,**kwargs):
        super(ImageWidget, self).__init__(*args,**kwargs)
        self._viewbox = self.addViewBox(row=1, col=1,enableMenu=False)
        if smart:
            self._img_item = SmartImageItem(255*np.ones((764,764)))
        else:
            self._img_item = ImageItem(np.zeros((764,764)))
        self._viewbox.addItem(self._img_item)
        self._viewbox.setAspectLocked(aspectLocked)
        self._viewbox.setMouseEnabled(*mouseEnabled)

        if isinstance(path,str):
            img = np.array(Image.open(path))
            self.setImage(img, levels=(0, 255))

    def imageItem(self):
        return self._img_item

    def viewBox(self):
        return self._viewbox

    def setImage(self,*args,**kwargs):
        if 'levels' not in kwargs.keys():
            kwargs['levels']=(0,255)
        self._img_item.setImage(*args,**kwargs)

class ControlImageWidget(QtWidgets.QWidget):
    def __init__(self,imageWidget,*args,**kwargs):
        super(ControlImageWidget,self).__init__(*args,**kwargs)

        self.imageWidget = imageWidget

        self.zoomInBtn = QtWidgets.QPushButton()
        self.zoomInBtn.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.zoomInBtn.setMaximumHeight(32)
        self.zoomInBtn.setIcon(Icon('zoom-in.svg'))

        self.zoomOutBtn = QtWidgets.QPushButton()
        self.zoomOutBtn.setMaximumHeight(32)
        self.zoomOutBtn.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.zoomOutBtn.setIcon(Icon('zoom-out.svg'))

        self.fullSizeBtn = QtWidgets.QPushButton()
        self.fullSizeBtn.setMaximumHeight(32)
        self.fullSizeBtn.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.fullSizeBtn.setIcon(Icon('maximize.svg'))

        self.panScaleBtn = QtWidgets.QPushButton()
        self.panScaleBtn.setMaximumHeight(32)
        self.panScaleBtn.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.panScaleBtn.setIcon(Icon('move.svg'))
        self.panScaleBtn.setCheckable(True)
        self.panScaleBtn.setChecked(True)

        self.zoomInBtn.clicked.connect(self.zoomIn)
        self.zoomOutBtn.clicked.connect(self.zoomOut)
        self.fullSizeBtn.clicked.connect(self.fullSize)
        self.panScaleBtn.toggled.connect(self.panScale)

        lspacer = QtWidgets.QWidget()
        lspacer.setSizePolicy(QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Preferred)
        rspacer = QtWidgets.QWidget()
        rspacer.setSizePolicy(QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Preferred)

        buttonLayout = QtGui.QGridLayout(self)
        buttonLayout.addWidget(lspacer,0,0)
        buttonLayout.addWidget(self.zoomInBtn,0,2)
        buttonLayout.addWidget(self.fullSizeBtn,0,3)
        buttonLayout.addWidget(self.zoomOutBtn,0,1)
        buttonLayout.addWidget(self.panScaleBtn,0,4)
        buttonLayout.addWidget(rspacer,0,5)
        buttonLayout.setAlignment(QtCore.Qt.AlignLeft)

        buttonLayout.setContentsMargins(0,0,0,0)

    def zoomIn(self):
        self.imageWidget.viewBox().scaleBy((0.9,0.9))

    def zoomOut(self):
        self.imageWidget.viewBox().scaleBy((1.1,1.1))

    def fullSize(self):
        self.imageWidget.viewBox().autoRange()

    def panScale(self,flag):
        self.imageWidget.viewBox().setMouseEnabled(flag,flag)

    def setImageWidget(self,widget):
        self.imageWidget = widget
        self.panScale(self.panScaleBtn.isChecked())



class ImageItem(pg.ImageItem):
    def setImage(self,image,*args,**kwargs):
        super(ImageItem,self).setImage(image[:,::-1,...],*args,**kwargs)

class DisplayWidgetFactory:
    def __init__(self,height=None,width=None):
        self.height = height
        self.width = width

    def __call__(self,*args,**kwargs):
        widget = QW.QWidget(*args,**kwargs)
        if isinstance(height,int):
            widget.setMinimumHeight(height)
        if isinstance(width,int):
            widget.setMinimumWidth(width)

class SmartImageItem(pg.ImageItem):
    imageUpdateSignal = QC.pyqtSignal(object,object)
    imageFinishSignal = QC.pyqtSignal()
    def __init__(self,*args,**kwargs):
        super(SmartImageItem,self).__init__(*args,**kwargs)
        self.base_cursor = self.cursor()
        self.radius = None
        self.clickOnly = False
        self.enableDraw = True

    def setImage(self,image,*args,**kwargs):
        super(SmartImageItem,self).setImage(image[:,::-1,...],*args,**kwargs)

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


HeaderLabel = LabelMaker(family='Helvetica',size=28,bold=True)
SubheaderLabel = LabelMaker(family='Helvetica',size=18)
BasicLabel = LabelMaker()

MaxSpacer = SpacerMaker()

StandardDisplay = DisplayWidgetFactory()
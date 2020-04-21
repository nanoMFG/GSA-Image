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

logger = logging.getLogger(__name__)

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
        listview.setUniformItemSizes(True)

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
    def __init__(self, path=None, parent=None):
        super(ImageWidget, self).__init__(parent=parent)
        self._viewbox = self.addViewBox(row=1, col=1)
        self._img_item = ImageItem()
        self._viewbox.addItem(self._img_item)
        self._viewbox.setAspectLocked(True)
        self._viewbox.setMouseEnabled(False,False)

        if isinstance(path,str):
            img = np.array(Image.open(path))
            self.setImage(img, levels=(0, 255))

        self._viewbox.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

    def imageItem(self):
        return self._img_item

    def viewBox(self):
        return self._viewbox

    def setImage(self,*args,**kwargs):
        self._img_item.setImage(*args,**kwargs)

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

def makeImageWidget():
    widget = pg.GraphicsLayoutWidget
    viewbox = widget.addViewBox(row=1, col=1)
    img_item = ImageItem()
    viewbox.addItem(img_item)
    viewbox.setAspectLocked(True)
    viewbox.setMouseEnabled(False,False)

    viewbox.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
    widget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

    return widget, img_item, viewbox

HeaderLabel = LabelMaker(family='Helvetica',size=28,bold=True)
SubheaderLabel = LabelMaker(family='Helvetica',size=18)
BasicLabel = LabelMaker()

MaxSpacer = SpacerMaker()

StandardDisplay = DisplayWidgetFactory(width=300)
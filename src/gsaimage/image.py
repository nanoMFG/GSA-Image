from __future__ import division

import io

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PyQt5 import QtGui, QtCore

from .gsaimage import FilterPattern, RemoveScale, Crop, DrawScale, InitialImage, Modification


class ImageEditor(QtGui.QScrollArea):
    submitClicked = QtCore.pyqtSignal(int,int,object) # sem_id, px_per_um, mask
    def __init__(self,sem_id,config,parent=None):
        super(ImageEditor,self).__init__(parent=parent)
        self.img = None
        self.sem_id = sem_id
        self.modifications = []
        self.config = config

        self.mod_dict = {
        'Filter Pattern': FilterPattern,
        'Draw Scale': DrawScale,
        'Crop': Crop,
        'Remove Scale': RemoveScale
        }

        self.wComboBox = pg.ComboBox()
        for item in sorted(list(self.mod_dict)):
            self.wComboBox.addItem(item)

        self.wAddMod = QtGui.QPushButton('Add')
        self.wAddMod.clicked.connect(lambda: self.addMod(mod=None))

        self.wRemoveMod = QtGui.QPushButton('Remove')
        self.wRemoveMod.clicked.connect(self.removeMod) 

        self.wReview = QtGui.QPushButton('Review')
        self.wReview.clicked.connect(self.review) 

        self.wModList = QtGui.QListWidget()
        self.wModList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.wModList.currentRowChanged.connect(self.selectMod)

        self.wDetail = QtGui.QStackedWidget()

        self.imgItem = pg.ImageItem()
        self.imgBox = pg.GraphicsLayoutWidget()
        self.imgBox_VB = self.imgBox.addViewBox(row=1,col=1)
        self.imgBox_VB.addItem(self.imgItem)
        self.imgBox_VB.setAspectLocked(True)

        self.contentWidget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout(self.contentWidget)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.setWidgetResizable(True)
        self.setWidget(self.contentWidget)

        self.layout.addWidget(self.wAddMod,0,0,1,1)
        self.layout.addWidget(self.wModList,1,0,1,1)
        self.layout.addWidget(self.wRemoveMod,2,0,1,1)
        self.layout.addWidget(self.wComboBox,3,0,1,1)
        self.layout.addWidget(self.wReview,4,0,1,1)
        self.layout.addWidget(self.imgBox,0,1,4,1)
        self.layout.addWidget(self.wDetail,0,2,4,1)


    def loadImage(self,data,thread_id,info):
        self._id = thread_id
        self.img = np.array(Image.open(io.BytesIO(data)).convert('L')).copy()

        mod = InitialImage(img_item=self.imgItem,properties={'mode':self.config.mode,'sem_id':self.sem_id})
        mod.set_image(self.img)
        self.addMod(mod)

        print(mod.image().shape)

    def addMod(self,mod=None):
        if mod == None:
            if len(self.modifications) > 0:
                mod = self.mod_dict[self.wComboBox.value()](self.modifications[-1],self.imgItem,properties={'mode':self.config.mode})
            else:
                return
        self.modifications.append(mod)
        self.wDetail.addWidget(mod.widget())
        self.wModList.addItem("%d %s"%(self.wModList.count(),mod.name()))
        if self.wModList.count() > 0:
            self.wModList.setCurrentRow(self.wModList.count()-1)

    def removeMod(self):
        if len(self.modifications) > 1:
            self.wDetail.removeWidget(self.modifications[-1].widget())
            self.selectMod(-1)
            del[self.modifications[-1]]
            self.wModList.takeItem(self.wModList.count()-1)
            if self.wModList.count() > 0:
                self.wModList.setCurrentRow(self.wModList.count()-1)

    def selectMod(self,index):
        if index >= 0:
            # try:
            self.modifications[index].update_view()
            # except:
                # pass
            self.wDetail.setCurrentIndex(index)
            print(index)
        elif self.wModList.count() > 0:
            self.wModList.setCurrentRow(self.wModList.count()-1)

    def review(self):
        mod = Review(
            self.modifications[-1],
            self.imgItem,
            properties={'mode':self.config.mode})
        mod.submitButton.clicked.connect(lambda: self.submitClicked.emit(*mod.update_image()))
        # mod.submitButton.clicked.connect(lambda: print(mod.update_image()))
        self.addMod(mod)


class Review(Modification):
    def __init__(self,mod_in=None,img_item=None,properties={}):
        super(Review,self).__init__(mod_in,img_item,properties)
        self.submitButton = QtGui.QPushButton('Submit')

        self.startImg = pg.ImageItem()
        self.startBox = pg.GraphicsLayoutWidget()
        self.startBox_VB = self.startBox.addViewBox(row=1,col=1)
        self.startBox_VB.addItem(self.startImg)
        self.startBox_VB.setAspectLocked(True)

        self.maskImg = pg.ImageItem()
        self.maskBox = pg.GraphicsLayoutWidget()
        self.maskBox_VB = self.maskBox.addViewBox(row=1,col=1)
        self.maskBox_VB.addItem(self.maskImg)
        self.maskBox_VB.setAspectLocked(True)

        self.px_per_um = QtGui.QLabel('')

        self.mainWidget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout(self.mainWidget)

        self.layout.addWidget(self.startBox,0,0,1,1)
        # self.layout.addWidget(self.maskBox,0,1,1,1)
        self.layout.addWidget(QtGui.QLabel('Pixels/um:'),1,0,1,1)
        self.layout.addWidget(self.px_per_um,1,1,1,1)
        self.layout.addWidget(self.submitButton,2,0,1,1)

    def update_image(self):
        start_img = self.root().image()
        final_img = start_img.copy()
        mask = None
        px_per_um = 0
        for mod in self.tolist():
            if mod.name() == 'Crop':
                if 'crop_coords' in mod.properties.keys():
                    crop_slice = np.array(mod.properties['crop_coords'])
                    final_img = final_img[crop_slice]
            elif mod.name() == 'Remove Scale':
                if 'scale_crop_box' in mod.properties.keys():
                    final_img = mod.crop_image(final_img,mod.properties['scale_crop_box'])
            elif mod.name() == 'Filter Pattern':
                if 'mask_total' in mod.properties.keys():
                    self.startImg.setImage(final_img.copy(),levels=(0,255))
                    mask = np.array(mod.properties['mask_total'])
                    final_img[np.logical_not(mask)] = 255 
                    self.maskImg.setImage(final_img,levels=(0,255))
            elif mod.name() == 'Draw Scale':
                if 'num_pixels' in mod.properties.keys():
                    px_per_um = mod.properties['num_pixels']
                    self.px_per_um.setText(str(px_per_um))

        return int(self.back_properties()['sem_id']), px_per_um, mask

    def name(self):
        return 'Review'

    def widget(self):
        return self.mainWidget

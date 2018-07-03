import numpy as np
import cv2
from PyQt5 import QtGui
import pyqtgraph as pg

class ImageAnalyzer:
  def __init__(self):
    self.img_file_path = None
    self.img_fname = None
    self.img_data = None
    self.modifications = None
    self.selectedWidget = None
    
    self.w = QtGui.QWidget()
    
    self.wOpenFileBtn = QtGui.QPushButton('Open Image')
    self.wOpenFileBtn.clicked.connect(self.openFile)    

    self.mod_dict = {'Color Mask': ColorMask}    
    
    self.wComboBox = pg.ComboBox()
    self.wComboBox.addItem('Color Mask')
    self.wComboBox.addItem('Binary Mask')
    self.wComboBox.addItem('Denoise')
    self.wComboBox.addItem('Canny Edge Detector')
    self.wComboBox.addItem('Find Contours')

    self.wAddMod = QtGui.QPushButton('Add Modification')
    self.wAddMod.clicked.connect(self.addMod)

    self.wRemoveMod = QtGui.QPushButton('Remove Modification')
    self.wRemoveMod.clicked.connect(self.removeMod)
    
    self.wModList = QtGui.QListWidget()
    self.wModList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
    self.wModList.itemClicked.connect(self.selectMod)
    
    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wImgItem = pg.ImageItem()
    self.wImgBox_VB.addItem(self.wImgItem)
    
    self.wImgBox_VB.addItem(self.wImgItem)
    self.wImgBox_VB.setAspectLocked(True)
    self.wImgBox_VB.setMouseEnabled(False,False)
    
    self.layout = QtGui.QGridLayout()
    self.layout.setColumnStretch(0,4)
    self.layout.addWidget(self.wOpenFileBtn, 0,0)
    self.layout.addWidget(self.wAddMod, 1,0)
    self.layout.addWidget(self.wRemoveMod,2,0)
    self.layout.addWidget(self.wComboBox, 3,0)
    self.layout.addWidget(self.wModList,4,0)
    self.layout.addWidget(self.wImgBox, 0,1,5,5)
    
    self.w.setLayout(self.layout)
  
  def updateAll(self,mod):
    mod.update_image()
    self.updateModWidget(mod.widget())
    self.updateListBox(mod.mod_list())
    if self.wModList.count() > 0:
      self.wModList.setCurrentRow(self.wModList.count()-1)

  def updateModWidget(self,widget):
    if self.selectedWidget != None:
      self.selectedWidget.hide()
    self.selectedWidget = widget
    if self.selectedWidget != None:
      self.selectedWidget.show()

  def updateListBox(self,mod_list):
    self.wModList.clear()
    for i,nm in enumerate(mod_list):
      self.wModList.addItem("%d %s"%(i,nm))

  def selectMod(self):
    selection = self.wModList.selectedItems()
    if len(selection) == 1:
      selection = selection[0]
      mod = self.modifications.back_traverse(self.wModList.count() - self.wModList.row(selection) - 1)
      mod.update_image()
      self.updateModWidget(mod.widget())

  def removeMod(self):
    selection = self.wModList.selectedItems()
    if len(selection) == 1:
      row = self.wModList.row(selection[0])
      if row == self.wModList.count() - 1:
        if self.modifications.widget() != None:
          self.modifications.widget().hide()
        self.modifications = self.modifications.mod_in
        self.updateAll(self.modifications)

  def addMod(self):
    value = self.wComboBox.value()
    if value == None:
      pass
    elif self.modifications != None:
      mod = self.mod_dict[value](self.modifications,self.wImgItem)
      widget = mod.widget()
      self.layout.addWidget(widget,6,1)
      self.updateAll(mod)
      self.modifications = mod
    
  def openFile(self):
    self.img_file_path = QtGui.QFileDialog.getOpenFileName()
    if isinstance(self.img_file_path,tuple):
      self.img_file_path = self.img_file_path[0]
    else:
      return
    self.img_fname = self.img_file_path.split('/')[-1]
    self.img_data = cv2.imread(self.img_file_path)
    self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_RGB2GRAY)
    
    mod = InitialImage(img_item=self.wImgItem)
    mod.set_image(self.img_data)

    self.updateAll(mod)
    self.modifications = mod

    self.w.setWindowTitle(self.img_fname)
  
  # def cannyEdgeDetection(self,low_thresh=None,high_thresh=None):
  #   if not isinstance(low_thresh,float):
  #     low_thresh = max(self.img_data.flatten())*.1
  #   if not isinstance(high_thresh,float):
  #     high_thresh = max(self.img_data.flatten())*.4
  #   self.img_state = cv2.GaussianBlur(self.img_state,(5,5),0)
  #   self.img_state = cv2.Canny(self.img_state,low_thresh,high_thresh)
  #   self.wImgItem.setImage(self.img_state,levels=(0,255))
  
  def run(self):
    self.w.show()
    app.exec_()

class Modification:
  def __init__(self,mod_in=None,img_item=None):
    self.mod_in = mod_in
    self.img_item = img_item
    if mod_in != None:
      self.img_out = self.mod_in.image()
    else:
      self.img_out = None

  def widget(self):
    pass  
  def image(self):
    return self.img_out
  def name(self):
    return 'Default Modification'
  def set_image(self,img):
    self.img_out = img
  def update_image(self):
    pass
  def delete_mod(self):
    return self.mod_in
  def mod_list(self):
    if self.mod_in != None:
      return self.mod_in.mod_list() + [self.name()]
    else:
      return [self.name()]
  def back_traverse(self,n):
    if n != 0:
      if self.mod_in == None:
        raise IndexError('Index out of range (n = %d)'%n)
      elif n != 0:
        return self.mod_in.back_traverse(n-1)
    elif n == 0:
      return self

class InitialImage(Modification):
  def widget(self):
    return None
  def update_image(self):
    self.img_item.setImage(self.img_out,levels=(0,255))
  def name(self):
    return 'Initial Image'

class ColorMask(Modification):
  def __init__(self,mod_in,img_item):
    super(ColorMask,self).__init__(mod_in,img_item)
    self.img_mask = None
    self.img_hist = self.img_item.getHistogram()

    self.wHistPlot = None
    self.lrItem = None

    self.wHistPlot = pg.PlotWidget()
    self.wHistPlot.plot(*self.img_hist)
    self.wHistPlot.setXRange(0,255)
    self.wHistPlot.hideAxis('left')
    
    self.lrItem = pg.LinearRegionItem((0,255),bounds=(0,255))
    self.lrItem.sigRegionChanged.connect(self.update_image)
    self.lrItem.sigRegionChangeFinished.connect(self.update_image)
    
    self.wHistPlot.addItem(self.lrItem)
    self.wHistPlot.setMouseEnabled(False,False)
    self.wHistPlot.setMaximumHeight(100)
  
  def widget(self):
    return self.wHistPlot

  def update_image(self):
    minVal, maxVal = self.lrItem.getRegion()
    img = self.mod_in.image()
    self.img_mask = np.zeros_like(img)
    self.img_mask[np.logical_and(img>minVal,img<maxVal)] = 1
    self.img_out = img*self.img_mask+(1-self.img_mask)*255
    self.img_item.setImage(self.img_out,levels=(0,255))

  def name(self):
    return 'Color Mask'    

class CannyEdgeDetection(Modification):
  def __init__(self,mod_in,img_item):
    super(CannyEdgeDetection,self).__init__(mod_in,img_item)
    self.wToolBox = pg.GraphicsLayoutWidget()
  def name(self):
    return 'Canny Edge Detection'
  def update_image(self):
    pass
  def widget(self):
    pass


app = QtGui.QApplication([])      
img_analyzer = ImageAnalyzer()
img_analyzer.run()
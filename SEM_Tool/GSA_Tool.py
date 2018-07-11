import numpy as np
import cv2
from PyQt5 import QtGui, QtCore
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

    self.mod_dict = {
    'Color Mask': ColorMask,
    'Canny Edge Detector': CannyEdgeDetection,
    'Dilate': Dilation,
    'Erode': Erosion,
    'Binary Mask': BinaryMask,
    'Find Contours': FindContours
    }    
    
    self.wComboBox = pg.ComboBox()
    self.wComboBox.addItem('Color Mask')
    self.wComboBox.addItem('Binary Mask')
    # self.wComboBox.addItem('Denoise')
    self.wComboBox.addItem('Canny Edge Detector')
    self.wComboBox.addItem('Dilate')
    self.wComboBox.addItem('Erode')
    self.wComboBox.addItem('Find Contours')

    self.wAddMod = QtGui.QPushButton('Add Modification')
    self.wAddMod.clicked.connect(self.addMod)

    self.wRemoveMod = QtGui.QPushButton('Remove Modification')
    self.wRemoveMod.clicked.connect(self.removeMod)
    
    self.wModList = QtGui.QListWidget()
    self.wModList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
    self.wModList.itemClicked.connect(self.selectMod)
    self.wModList.itemSelectionChanged.connect(self.selectMod)
    
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
      if self.wModList.row(selection) < self.wModList.count() - 1:
        self.updateModWidget(None)
      else:
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
      self.modifications = mod
      widget = mod.widget()
      self.layout.addWidget(widget,6,1)
      self.updateAll(mod)
    
  def openFile(self):
    try:
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
      self.modifications = mod
      
      self.updateAll(mod)
      self.w.setWindowTitle(self.img_fname)
    except:
      return
  
  def run(self):
    self.w.show()
    app.exec_()

class Modification:
  def __init__(self,mod_in=None,img_item=None):
    self.mod_in = mod_in
    self.img_item = img_item
    self.scale = 1
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
  def root(self):
    if self.mod_in != None:
      return self.mod_in.root()
    else:
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
    
    self.low_thresh = int(max(self.mod_in.image().flatten())*.1)
    self.high_thresh = int(max(self.mod_in.image().flatten())*.4)
    self.gauss_size = 5
    self.wToolBox = pg.LayoutWidget()

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
    self.wToolBox.addWidget(QtGui.QLabel('High Threshold'),2,0)
    self.wToolBox.addWidget(self.wGaussEdit,0,1)
    self.wToolBox.addWidget(self.wLowEdit,1,1)
    self.wToolBox.addWidget(self.wHighEdit,2,1)
    self.wToolBox.addWidget(self.wLowSlider,1,2)
    self.wToolBox.addWidget(self.wHighSlider,2,2)

  def name(self):
    return 'Canny Edge Detection'

  def _update_sliders(self):
    self.gauss_size = int(self.wGaussEdit.text())
    self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
    self.wGaussEdit.setText(str(self.gauss_size))
    self.low_thresh = int(self.wLowEdit.text())
    self.high_thresh = int(self.wHighEdit.text())

    self.wHighSlider.setSliderPosition(self.low_thresh)    
    self.wHighSlider.setSliderPosition(self.high_thresh)

    self.update_image()

  def _update_texts(self):
    self.low_thresh = int(self.wLowSlider.value())
    self.high_thresh = int(self.wHighSlider.value())

    self.wLowEdit.setText(str(self.low_thresh))
    self.wHighEdit.setText(str(self.high_thresh))

    self.update_image()

  def update_image(self):
    self.img_out = cv2.GaussianBlur(self.mod_in.image(),(self.gauss_size,self.gauss_size),0)
    self.img_out = 255-cv2.Canny(self.img_out,self.low_thresh,self.high_thresh)
    self.img_item.setImage(self.img_out,levels=(0,255))
  
  def widget(self):
    return self.wToolBox

class Dilation(Modification):
  def __init__(self,mod_in,img_item):
    super(Dilation,self).__init__(mod_in,img_item)
    self.size = 1
    self.wToolBox = pg.LayoutWidget()
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
    self.wToolBox.addWidget(self.wSizeSlider,0,2)

  def name(self):
    return 'Dilation'

  def _update_sliders(self):
    self.size = int(self.wSizeEdit.text())
    self.wSizeSlider.setSliderPosition(self.size)
    self.update_image()

  def _update_texts(self):
    self.size = int(self.wSizeSlider.value())
    self.wSizeEdit.setText(str(self.size))
    self.update_image()

  def update_image(self):
    self.img_out = cv2.erode(self.mod_in.image().copy(),np.ones((self.size,self.size),np.uint8),iterations=1)
    self.img_item.setImage(self.img_out,levels=(0,255))

  def widget(self):
    return self.wToolBox

class Erosion(Modification):
  def __init__(self,mod_in,img_item):
    super(Erosion,self).__init__(mod_in,img_item)
    self.size = 1
    self.wToolBox = pg.LayoutWidget()
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
    self.wToolBox.addWidget(self.wSizeSlider,0,2)

  def name(self):
    return 'Erosion'

  def _update_sliders(self):
    self.size = int(self.wSizeEdit.text())
    self.wSizeSlider.setSliderPosition(self.size)
    self.update_image()

  def _update_texts(self):
    self.size = int(self.wSizeSlider.value())
    self.wSizeEdit.setText(str(self.size))
    self.update_image()

  def update_image(self):
    self.img_out = cv2.dilate(self.mod_in.image(),np.ones((self.size,self.size),np.uint8),iterations=1)
    self.img_item.setImage(self.img_out,levels=(0,255))

  def widget(self):
    return self.wToolBox

class BinaryMask(Modification):
  def __init__(self,mod_in,img_item):
    super(BinaryMask,self).__init__(mod_in,img_item)

  def name(self):
    return 'Binary Mask'

  def update_image(self):
    self.img_out = self.mod_in.image()
    self.img_out[self.img_out < 255] = 0
    self.img_item.setImage(self.img_out,levels=(0,255))

  def widget(self):
    return None

class FindContours(Modification):
  def __init__(self,mod_in,img_item):
    super(FindContours,self).__init__(mod_in,img_item)
    self.wLayout = pg.LayoutWidget()
    self.img_inv = self.mod_in.image().copy()
    self.img_inv[self.img_inv < 255] = 0
    self.img_inv = 255 - self.img_inv

    self.tol = 0.04
    self.wTolEdit = QtGui.QLineEdit(str(self.tol))
    self.wTolEdit.setValidator(QtGui.QDoubleValidator(0,1,3))
    self.wTolEdit.setFixedWidth(60)

    self.lowVert = 6
    self.wLowEdit = QtGui.QLineEdit(str(self.lowVert))
    self.wLowEdit.setValidator(QtGui.QIntValidator(3,20))
    self.wLowEdit.setFixedWidth(60)

    self.highVert = 6
    self.wHighEdit = QtGui.QLineEdit(str(self.highVert))
    self.wHighEdit.setValidator(QtGui.QIntValidator(3,20))
    self.wHighEdit.setFixedWidth(60)

    self._img, self.contours, self.hierarchy = cv2.findContours(
      self.img_inv,
      cv2.RETR_LIST,
      cv2.CHAIN_APPROX_SIMPLE)

    self.contour_dict = {}
    
    self.wContourList = QtGui.QListWidget()
    self.wContourList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
    for i,cnt in enumerate(self.contours):
      key = '%d Contour'%i
      self.contour_dict[key] = {}
      self.contour_dict[key]['index'] = i
      self.contour_dict[key]['list_item'] = QtGui.QListWidgetItem(key)
      self.contour_dict[key]['contour'] = cnt
      self.contour_dict[key]['area'] = cv2.contourArea(cnt,oriented=True)
    self.update_tol()

    self.wContourList.itemSelectionChanged.connect(self.update_image)
    self.wContourList.itemClicked.connect(self.update_image)
    self.wTolEdit.returnPressed.connect(self.update_tol)
    self.wLowEdit.returnPressed.connect(self.update_tol)
    self.wHighEdit.returnPressed.connect(self.update_tol)
    
    if len(self.contour_dict.keys())>0:
      self.wContourList.setCurrentItem(self.contour_dict['0 Contour']['list_item'])
    
    self.wLayout.addWidget(self.wContourList,0,0)
    self.wLayout.addWidget(QtGui.QLabel('Polygon Tolerance:'),0,1)
    self.wLayout.addWidget(self.wTolEdit,0,2)
    self.wLayout.addWidget(QtGui.QLabel('Vertex Tolerance:'),1,1)
    self.wLayout.addWidget(self.wLowEdit,1,2)
    self.wLayout.addWidget(self.wHighEdit,1,3)

    self.update_image()

  def update_image(self):
    self.img_out = self.mod_in.image().copy()
    selection = self.wContourList.selectedItems()
    if len(selection) == 1:
      cnt_key = selection[0].text()
      cv2.drawContours(self.img_out,self.contours,self.contour_dict[cnt_key]['index'],thickness=-1,color=(0,255,0))
    self.img_item.setImage(self.img_out,levels=(0,255))

  def detect_poly(self,cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, self.tol * peri, True)
    return len(approx) >= self.lowVert and len(approx) <= self.highVert

  def update_tol(self):
    self.tol = float(self.wTolEdit.text())
    self.lowVert = float(self.wLowEdit.text())
    self.highVert = float(self.wHighEdit.text())
    self.wContourList.clear()
    for key in self.contour_dict.keys():
      cnt = self.contour_dict[key]['contour']
      area = self.contour_dict[key]['area']
      if self.detect_poly(cnt) and  area < 0:
        self.contour_dict[key]['list_item'] = QtGui.QListWidgetItem(key)
        self.wContourList.addItem(self.contour_dict[key]['list_item'])

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Find Contours'

def FilterPattern(Modification):
  def __init__(self,mod_in,img_item):
    super(FindContours,self).__init__(mod_in,img_item)

app = QtGui.QApplication([])      
img_analyzer = ImageAnalyzer()
img_analyzer.run()
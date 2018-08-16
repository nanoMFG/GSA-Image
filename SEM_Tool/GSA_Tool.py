from __future__ import division
import numpy as np
import cv2, sys, time
from skimage import transform
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg

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

class ImageAnalyzer:
  def __init__(self):
    self.img_file_path = None
    self.img_fname = None
    self.img_data = None
    self.modifications = None
    self.properties = {}
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
    'Find Contours': FindContours,
    'Filter Pattern': FilterPattern,
    'Blur': Blur,
    'Draw Scale': DrawScale,
    'Crop': Crop,
    'Domain Centers': DomainCenters,
    'Hough Transform': HoughTransform
    }    
    
    self.wComboBox = pg.ComboBox()
    for item in sorted(list(self.mod_dict)):
      self.wComboBox.addItem(item)  

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
    self.wImgBox_VB.setAspectLocked(True)
    # self.wImgBox_VB.setMouseEnabled(False,False)
    
    self.layout = QtGui.QGridLayout()
    self.layout.setColumnStretch(0,10)
    self.layout.addWidget(self.wOpenFileBtn, 0,0)
    self.layout.addWidget(self.wAddMod, 1,0)
    self.layout.addWidget(self.wRemoveMod,2,0)
    self.layout.addWidget(self.wComboBox, 3,0)
    self.layout.addWidget(self.wModList,4,0)
    self.layout.addWidget(self.wImgBox, 0,1,5,5)
    
    self.w.setLayout(self.layout)
  
  def updateAll(self,mod):
    try:
      self.properties.update(mod.update_image())
    except:
      pass
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
      try:
        self.properties.update(mod.update_image())
      except:
        pass
      if self.wModList.row(selection) < self.wModList.count() - 1:
        self.updateModWidget(None)
      else:
        self.updateModWidget(mod.widget())

  def removeMod(self):
    selection = self.wModList.selectedItems()
    if len(selection) == 1 and self.modifications != None:
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
      self.layout.addWidget(widget,0,7,5,5,alignment=QtCore.Qt.AlignTop)
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
  def __init__(self,mod_in=None,img_item=None,properties={}):
    self.mod_in = mod_in
    self.img_item = img_item
    self.properties = properties
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
    return self.properties
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
    return self.properties
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

    return self.properties

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
    self.wToolBox.addWidget(QtGui.QLabel('High Threshold'),3,0)
    self.wToolBox.addWidget(self.wGaussEdit,0,1)
    self.wToolBox.addWidget(self.wLowEdit,1,1)
    self.wToolBox.addWidget(self.wHighEdit,3,1)
    self.wToolBox.addWidget(self.wLowSlider,2,0,1,2)
    self.wToolBox.addWidget(self.wHighSlider,4,0,1,2)

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
    self.img_out = 255-cv2.Canny(self.img_out,self.low_thresh,self.high_thresh,L2gradient=True)
    self.img_item.setImage(self.img_out,levels=(0,255))

    return self.properties
  
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
    self.wToolBox.addWidget(self.wSizeSlider,1,0,1,2)

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

    return self.properties

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
    self.wToolBox.addWidget(self.wSizeSlider,1,0,1,2)

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

    return self.properties

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

    return self.properties

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
    self.wLowEdit.setValidator(QtGui.QIntValidator(3,100))
    self.wLowEdit.setFixedWidth(60)

    self.highVert = 6
    self.wHighEdit = QtGui.QLineEdit(str(self.highVert))
    self.wHighEdit.setValidator(QtGui.QIntValidator(3,100))
    self.wHighEdit.setFixedWidth(60)

    self.wThreshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wThreshSlider.setMinimum(0)
    self.wThreshSlider.setMaximum(100)
    self.wThreshSlider.setSliderPosition(50)

    self._img, self.contours, self.hierarchy = cv2.findContours(
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

    self.wContourList.itemSelectionChanged.connect(self.update_image)
    self.wContourList.itemClicked.connect(self.update_image)
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

    self.update_image()

  def update_image(self):
    self.img_out = self.mod_in.image().copy()
    selection = self.wContourList.selectedItems()
    if len(selection) == 1:
      cnt_key = selection[0].text()
      accept, approx = self.detect_poly(self.contour_dict[cnt_key]['contour'])
      cv2.drawContours(self.img_out,[approx],0,thickness=2,color=(0,255,0))
    self.img_item.setImage(self.img_out,levels=(0,255))

    return self.properties

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
  def __init__(self,mod_in,img_item):
    super(Blur,self).__init__(mod_in,img_item)
    self.gauss_size = 5
    self.wLayout = pg.LayoutWidget()
    self.wGaussEdit = QtGui.QLineEdit(str(self.gauss_size))
    self.wGaussEdit.setValidator(QtGui.QIntValidator(3,51))
    self.wGaussEdit.setFixedWidth(60)   

    self.wLayout.addWidget(QtGui.QLabel('Gaussian Size'),0,0)
    self.wLayout.addWidget(self.wGaussEdit,1,0)

    self.update_image()
    self.wGaussEdit.returnPressed.connect(self.update_image)

  def update_image(self):
    self.gauss_size = int(self.wGaussEdit.text())
    self.gauss_size = self.gauss_size + 1 if self.gauss_size % 2 == 0 else self.gauss_size
    self.wGaussEdit.setText(str(self.gauss_size))

    self.img_out = cv2.GaussianBlur(self.mod_in.image(),(self.gauss_size,self.gauss_size),0)
    self.img_item.setImage(self.img_out,levels=(0,255))

    return self.properties

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Blur'

class FilterPattern(Modification):
  def __init__(self,mod_in,img_item):
    super(FilterPattern,self).__init__(mod_in,img_item)
    self.roi_size = 20
    self.scale = 1
    self.roi_list = []
    self.roi_item = None

    self.wLayout = pg.LayoutWidget()

    self.wFilterList = QtGui.QListWidget()
    self.wFilterList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
    self.wAdd = QtGui.QPushButton('Add Filter')
    self.wRemove = QtGui.QPushButton('Remove Filter')
    self.wErase = QtGui.QPushButton('Erase')

    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wImgROI = pg.ImageItem()
    self.wImgROI.setImage(self.img_item.image,levels=(0,255))
    self.wImgBox_VB.addItem(self.wImgROI)
    self.wImgBox_VB.setAspectLocked(True)
    # self.wImgBox_VB.setMouseEnabled(False,False)

    self.wComboBox = pg.ComboBox()
    self.wComboBox.addItem('TM_SQDIFF')
    self.wComboBox.addItem('TM_SQDIFF_NORMED')
    self.wComboBox.addItem('TM_CCORR')
    self.wComboBox.addItem('TM_CCORR_NORMED')
    self.wComboBox.addItem('TM_CCOEFF')
    self.wComboBox.addItem('TM_CCOEFF_NORMED')

    self.method_dict = {
      'TM_SQDIFF': cv2.TM_SQDIFF,
      'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
      'TM_CCORR': cv2.TM_CCORR,
      'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
      'TM_CCOEFF': cv2.TM_CCOEFF,
      'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED
    }

    self.wThreshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wThreshSlider.setMinimum(1)
    self.wThreshSlider.setMaximum(1000)
    self.wThreshSlider.setSliderPosition(100)

    self.wSizeSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wSizeSlider.setMinimum(2)
    self.wSizeSlider.setMaximum(50)
    self.wSizeSlider.setSliderPosition(5)

    self.wFilterAreaLabel = QtGui.QLabel('')

    self.wLayout.addWidget(QtGui.QLabel('Filter Method:'),0,0)
    self.wLayout.addWidget(self.wComboBox,0,1,1,2)
    self.wLayout.addWidget(QtGui.QLabel('Threshold:'),1,0)
    self.wLayout.addWidget(self.wThreshSlider,1,1,1,3)
    self.wLayout.addWidget(QtGui.QLabel('ROI Size:'),2,0)
    self.wLayout.addWidget(self.wSizeSlider,2,1,1,3)
    self.wLayout.addWidget(QtGui.QLabel('Filter Area:'),3,0)
    self.wLayout.addWidget(self.wFilterAreaLabel,3,1)
    self.wLayout.addWidget(self.wImgBox,7,0,4,4)
    self.wLayout.addWidget(self.wFilterList,4,0,3,3)
    self.wLayout.addWidget(self.wAdd,4,3)
    self.wLayout.addWidget(self.wRemove,5,3)
    self.wLayout.addWidget(self.wErase,6,3)

    self.wThreshSlider.valueChanged.connect(self.update_image)
    self.wComboBox.currentIndexChanged.connect(self.update_image)
    self.wSizeSlider.valueChanged.connect(self.update_image)
    self.wAdd.clicked.connect(self.addROI)
    self.wRemove.clicked.connect(self.removeROI)
    self.wFilterList.itemSelectionChanged.connect(self.selectROI)

  def addROI(self):
    roi = pg.ROI(
      pos=(0,0),
      size=(self.roi_size,self.roi_size),
      removable=True,
      pen=pg.mkPen(color='r',width=2),
      maxBounds=self.wImgROI.boundingRect(),
      scaleSnap=True,
      snapSize=2)
    roi.sigRegionChanged.connect(self.update_image)
    self.wImgBox_VB.addItem(roi)
    if len(self.roi_list) == 0:
      image_in = self.mod_in.image()
    else:
      image_in = self.roi_list[-1]['image_out']
    roi_dict = {'roi':roi,'threshold':100,'image_in':image_in.copy(),'image_out':image_in.copy()}
    self.roi_list.append(roi_dict)
    self.wFilterList.addItem('Filter %d'%self.wFilterList.count())
    self.wFilterList.setCurrentRow(self.wFilterList.count()-1)
    self.selectROI()

  def removeROI(self):
    self.wImgBox_VB.removeItem(self.roi_list[-1]['roi'])
    del self.roi_list[-1]
    self.wFilterList.takeItem(self.wFilterList.count()-1)
    self.wFilterList.setCurrentRow(self.wFilterList.count()-1)
    self.selectROI()

  def selectROI(self):
    selection = self.wFilterList.selectedItems()
    if len(selection) == 1:
      selection = selection[0]
      row = self.wFilterList.row(selection)
      self.roi_item = self.roi_list[row]
      self.wImgROI.setImage(self.roi_item['image_in'],levels=(0,255))
      roi = self.roi_item['roi']
      self.wSizeSlider.setSliderPosition(int(roi.size()[0]/2))
      self.wThreshSlider.setSliderPosition(self.roi_item['threshold'])

      for r,ro in enumerate(self.roi_list):
        if r != row:
          ro['roi'].hide()
        else:
          ro['roi'].show()
    else:
      self.roi_item = None

    self.update_image()

  def update_image(self):
    try:
      if self.roi_item != None:
        roi = self.roi_item['roi']
        if 2*int(self.wSizeSlider.value()) != roi.size()[0]:
          roi_size = 2*int(self.wSizeSlider.value())
          roi.setSize([roi_size,roi_size])
        
        value = self.wComboBox.value()
        region = roi.getArrayRegion(self.mod_in.image(),self.wImgROI)
        region = region.astype(np.uint8)

        self.roi_item['mask_idxs'] = np.zeros_like(self.mod_in.image()).astype(bool)

        for i in range(4):
          region = np.rot90(region,k=i)
          x,y = region.shape
          padded_image = cv2.copyMakeBorder(self.mod_in.image().copy(),int(y/2-1),int(y/2),int(x/2-1),int(x/2),cv2.BORDER_REFLECT_101)
          res = cv2.matchTemplate(padded_image,region,self.method_dict[value])
          
          maxVal = res.flatten().max()
          threshold = maxVal * np.logspace(-3,0,1000)[self.wThreshSlider.value()]
          self.roi_item['mask_idxs'][res < threshold] = True
        
        self.img_out = self.roi_item['image_in'].copy()
        self.img_out[self.roi_item['mask_idxs']] = 255
        self.roi_item['image_out'] = self.img_out.copy()
        self.wFilterAreaLabel.setNum(np.sum(self.roi_item['mask_idxs'])*self.scale**2)
        self.img_item.setImage(self.img_out,levels=(0,255))

        roi_img = self.mod_in.image().copy()
        roi_img = cv2.cvtColor(roi_img,cv2.COLOR_GRAY2BGR)
        for r in self.roi_list:
          roi_img[r['mask_idxs'],2] = 255
        self.wImgROI.setImage(roi_img)

    except Exception as e:
      print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    return self.properties


  def widget(self):
    return self.wLayout

  def name(self):
    return 'Filter Pattern'

class Crop(Modification):
  def __init__(self,mod_in,img_item):
    super(Crop,self).__init__(mod_in,img_item)
    self.roi_size = 20
    self.wLayout = pg.LayoutWidget()

    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wImgROI = pg.ImageItem()
    self.wImgROI.setImage(self.img_item.image,levels=(0,255))
    self.wImgBox_VB.addItem(self.wImgROI)
    self.wImgBox_VB.setAspectLocked(True)
    self.wImgBox_VB.setMouseEnabled(False,False)

    self.roi = pg.ROI(
      pos=(0,0),
      size=(self.roi_size,self.roi_size),
      removable=True,
      pen=pg.mkPen(color='r',width=2),
      maxBounds=self.wImgROI.boundingRect(),
      scaleSnap=True,
      snapSize=2)
    self.roi.addScaleHandle(pos=(1,1),center=(0,0),lockAspect=True)
    self.wImgBox_VB.addItem(self.roi)
    self.roi.sigRegionChanged.connect(self.update_image)

    self.wLayout.addWidget(self.wImgBox,0,0)

  def update_image(self):
    self.img_out = self.roi.getArrayRegion(self.wImgROI.image,self.wImgROI)
    self.img_out = self.img_out.astype(np.uint8)
    self.img_item.setImage(self.img_out,levels=(0,255))

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Crop'

class HoughTransform(Modification):
  def __init__(self,mod_in,img_item):
    super(HoughTransform,self).__init__(mod_in,img_item)
    self.inv_img = 255 - self.mod_in.image()
    self.img_out = self.mod_in.image().copy()
    self.wLayout = pg.LayoutWidget()
    self.hspace,self.angles,self.distances = transform.hough_line(self.inv_img)

    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wHough = pg.ImageItem()
    self.wHough.setImage(self.hspace,levels=(0,255))
    self.wImgBox_VB.addItem(self.wHough)
    self.wImgBox_VB.setAspectLocked(True)
    self.wImgBox_VB.setMouseEnabled(False,False)

    self.wMinAngleSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wMinAngleSlider.setMinimum(5)
    self.wMinAngleSlider.setMaximum(180)
    self.wMinAngleSlider.setSliderPosition(10)
    
    self.wMinDistSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wMinDistSlider.setMinimum(5)
    self.wMinDistSlider.setMaximum(200)
    self.wMinDistSlider.setSliderPosition(9)

    self.wThreshSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wThreshSlider.setMinimum(0)
    self.wThreshSlider.setMaximum(200)
    self.wThreshSlider.setSliderPosition(100)

    self.wLengthSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wLengthSlider.setMinimum(10)
    self.wLengthSlider.setMaximum(200)
    self.wLengthSlider.setSliderPosition(50)
    
    self.wGapSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
    self.wGapSlider.setMinimum(5)
    self.wGapSlider.setMaximum(100)
    self.wGapSlider.setSliderPosition(10)

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

    self.wThreshSlider.valueChanged.connect(self.update_image)
    self.wMinAngleSlider.valueChanged.connect(self.update_image)
    self.wMinDistSlider.valueChanged.connect(self.update_image)
    self.wLengthSlider.valueChanged.connect(self.update_image)
    self.wGapSlider.valueChanged.connect(self.update_image)

  def update_image(self):
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

    lines = transform.probabilistic_hough_line(
      self.inv_img,
      threshold=self.threshold,
      line_length=self.line_length,
      line_gap=self.line_gap)

    bgr_hough = np.round(self.hspace/np.max(self.hspace)*255).astype(np.uint8)
    bgr_hough = cv2.cvtColor(bgr_hough,cv2.COLOR_GRAY2BGR)

    for a,d in zip(angles,dists):
      angle_idx = np.nonzero(a == self.angles)[0]
      dist_idx = np.nonzero(d == self.distances)[0]
      cv2.circle(bgr_hough,center=(angle_idx,dist_idx),radius=5,color=(0,0,255),thickness=-1)

    self.wHough.setImage(bgr_hough)
    
    bgr_img = self.mod_in.image().copy()
    bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_GRAY2BGR)

    for p1,p2 in lines:
      cv2.line(bgr_img,p1,p2,color=(0,0,255),thickness=2)

    self.img_item.setImage(bgr_img)

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Hough Transform'

class DomainCenters(Modification):
  def __init__(self,mod_in,img_item):
    super(DomainCenters,self).__init__(mod_in,img_item)
    self.roi_list = []
    self.properties['centers'] = []
    self.wLayout = pg.LayoutWidget()

    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wImgROI = pg.ImageItem()
    self.wImgROI.setImage(self.img_item.image,levels=(0,255))
    self.wImgBox_VB.addItem(self.wImgROI)
    self.wImgBox_VB.setAspectLocked(True)
    self.wImgBox_VB.setMouseEnabled(False,False)

    self.wLayout.addWidget(self.wImgBox,0,0)
    self.wImgBox.scene().sigMouseClicked.connect(self.update_image)

  def update_image(self):
    pos = self.wImgBox_VB.mapSceneToView(self.wImgBox.lastMousePos)

    roi = pg.CircleROI(pos,
      [10,10],
      pen=pg.mkPen(color='r',width=2),
      maxBounds=self.wImgBox.boundingRect())
    self.wImgBox_VB.addItem(roi)
    handles = roi.getHandles()
    for h in handles:
      roi.removeHandle(h)
    self.roi_list.append(roi)
    print(self.wImgBox_VB.screenGeometry(),self.wImgBox_VB.mapFromViewToItem(self.wImgROI,pos))

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Domain Centers'

class DrawScale(Modification):
  def __init__(self,mod_in,img_item):
    super(DrawScale,self).__init__(mod_in,img_item)
    self.num_pixels = 1
    self.length = 1
    self.scale = 1
    
    self.wLayout = pg.LayoutWidget()

    self.wImgBox = pg.GraphicsLayoutWidget()
    self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
    self.wImgROI = pg.ImageItem()
    self.wImgROI.setImage(self.img_item.image,levels=(0,255))
    self.wImgBox_VB.addItem(self.wImgROI)
    self.wImgBox_VB.setAspectLocked(True)
    self.wImgBox_VB.setMouseEnabled(False,False)

    self.wPixels = QtGui.QLabel('1')
    self.wPixels.setFixedWidth(60)
    self.wScale = QtGui.QLabel('1')
    self.wScale.setFixedWidth(60)

    self.wLengthEdit = QtGui.QLineEdit(str(self.scale))
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

    self.roi.sigRegionChanged.connect(self.update_image)
    self.wLengthEdit.returnPressed.connect(self.update_image)

  def update_image(self):
    self.num_pixels = len(self.roi.getArrayRegion(self.mod_in.image(),self.img_item))
    self.wPixels.setNum(self.num_pixels)
    self.length = float(self.wLengthEdit.text())
    self.scale = self.length / self.num_pixels
    self.wScale.setText(str(self.scale))
    self.properties['scale'] = self.scale

    return self.properties

  def widget(self):
    return self.wLayout

  def name(self):
    return 'Draw Scale'


app = QtGui.QApplication([])      
img_analyzer = ImageAnalyzer()
img_analyzer.run()
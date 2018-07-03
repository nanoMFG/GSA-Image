import numpy as np
import cv2, os
from PyQt5 import QtGui
import pyqtgraph as pg


class HexagonalSegmentation:
	def __init__(self):
		self.img_file_path = None
		self.img_fname = None
		self.img_data = None
		
		self.w = QtGui.QWidget()

		self.wOpenFileBtn = QtGui.QPushButton('Open Image')
		self.wOpenFileBtn.clicked.connect(self.openFile)    

		self.roi_dict = {}

		self.wROIList = QtGui.QListWidget()
		self.wROIList.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
		self.wROIList.itemClicked.connect(self.listChanged)

		self.wAddROIBtn = QtGui.QPushButton('Add ROI')
		self.wAddROIBtn.clicked.connect(self.addROI)

		self.wExportROIBtn = QtGui.QPushButton('Export all ROI')
		self.wExportROIBtn.clicked.connect(self.exportROIList)

		self.wRemoveROIBtn = QtGui.QPushButton('Remove ROI')
		self.wRemoveROIBtn.clicked.connect(self.removeROI)

		self.wImgBox = pg.GraphicsLayoutWidget()
		self.wImgBox_VB = self.wImgBox.addViewBox(row=1,col=1)
		self.wImgItem = pg.ImageItem()
		self.wImgBox_VB.addItem(self.wImgItem)
		self.wImgBox_VB.setAspectLocked(True)

		self.wZoomBox = pg.GraphicsLayoutWidget()
		self.wZoomBox_VB = self.wZoomBox.addViewBox(row=1,col=1)
		self.wZoomItem = pg.ImageItem()
		self.wZoomBox_VB.addItem(self.wZoomItem)
		self.wZoomBox_VB.setAspectLocked(True)
		self.wZoomBox_VB.setMouseEnabled(False,False)

		self.wZoomBox_VB.addItem(pg.GridItem())

		self.wRotCW = QtGui.QPushButton('Rotate CW')
		self.wRotCW.clicked.connect(lambda: self.rotate(self.wAngle.text(),-1))
		self.wRotCCW = QtGui.QPushButton('Rotate CCW')
		self.wRotCCW.clicked.connect(lambda: self.rotate(self.wAngle.text(),1))
		self.wAngle = QtGui.QLineEdit('0')
		self.wAngle.returnPressed.connect(lambda: self.rotate(self.wAngle.text(),0))

		double_validator = QtGui.QDoubleValidator()
		self.wAngle.setValidator(double_validator)

		self.layout = QtGui.QGridLayout()
		for i in range(0,3):
			self.layout.setColumnStretch(i,3)
		self.layout.addWidget(self.wOpenFileBtn, 0,0)
		self.layout.addWidget(self.wAddROIBtn, 0,1)
		self.layout.addWidget(self.wRemoveROIBtn, 0,2)
		self.layout.addWidget(self.wROIList, 1,0,1,3)
		self.layout.addWidget(self.wExportROIBtn,2,2)
		self.layout.addWidget(self.wRotCW, 3,0)
		self.layout.addWidget(self.wRotCCW, 3,2)
		self.layout.addWidget(self.wAngle,3,1)
		self.layout.addWidget(self.wImgBox, 0,4,7,6)
		self.layout.addWidget(self.wZoomBox,4,0,4,3)

		self.w.setLayout(self.layout)

	def openFile(self):
		self.img_file_path = QtGui.QFileDialog.getOpenFileName()
		if isinstance(self.img_file_path,tuple):
			self.img_file_path = self.img_file_path[0]
		else:
			return
		self.img_fname = self.img_file_path.split('/')[-1]
		self.img_data = cv2.imread(self.img_file_path)
		self.img_data = cv2.cvtColor(self.img_data, cv2.COLOR_RGB2GRAY)
		self.wImgItem.setImage(self.img_data,levels=(0,255))

		self.w.setWindowTitle(self.img_fname)

	def removeROI(self):
		selection = self.wROIList.selectedItems()
		if len(selection) == 1:
			roi_name = selection[0].text()
			roi, list_item = self.roi_dict[roi_name]
			self.wROIList.takeItem(self.wROIList.row(list_item))
			self.wImgBox_VB.removeItem(roi)
			del self.roi_dict[roi_name]

	def addROI(self):
		roi_name = 'ROI_%d'%self.wROIList.count()

		new_roi = pg.ROI(pos=(0,0),size=(100,100),removable=True)
		new_roi.addScaleHandle(pos=(1,1),center=(0,0),lockAspect=True)
		self.wImgBox_VB.addItem(new_roi)
		list_item = QtGui.QListWidgetItem(roi_name)
		self.roi_dict[roi_name] = [new_roi,list_item]
		self.wROIList.addItem(list_item)
		self.wROIList.setCurrentRow(self.wROIList.count()-1)
		new_roi.sigClicked.connect(lambda: self.selectROI(roi_name))
		new_roi.sigRegionChanged.connect(lambda: self.selectROI(roi_name))
		self.selectROI(roi_name)

	def listChanged(self):
		selection = self.wROIList.selectedItems()
		if len(selection) == 1:
			self.selectROI(selection[0].text())

	def selectROI(self,roi_name):
		roi = self.roi_dict[roi_name][0]
		list_item = self.roi_dict[roi_name][1]
		self.wROIList.setCurrentItem(list_item)
		region = roi.getArrayRegion(self.img_data,self.wImgItem)
		self.wAngle.setText(str(roi.angle()))
		self.updateZoomBox(region)
		
	def updateZoomBox(self,region):
		self.wZoomItem.setImage(region,levels=(0,255))

	def rotate(self,angle,inc=0):
		selection = self.wROIList.selectedItems()
		if len(selection) == 1:
			roi = self.roi_dict[selection[0].text()][0]
			if angle == '':
				angle = 0
			else:
				angle = float(angle)
			roi.setAngle((angle+inc)%360)

	def exportROIList(self):
		save_directory = QtGui.QFileDialog.getExistingDirectory()
		
		for roi_name in self.roi_dict.keys():
			roi = self.roi_dict[roi_name][0]
			region = roi.getArrayRegion(self.img_data,self.wImgItem)
			cv2.imwrite(os.path.join(save_directory,roi_name)+'.png',region)

	def run(self):
		self.w.show()
		app.exec_()


app = QtGui.QApplication([])
hexseg = HexagonalSegmentation()
hexseg.run()
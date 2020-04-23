from PIL import Image
from PyQt5 import QtGui
import logging
import cairosvg, io, os

logger = logging.getLogger(__name__)

QG = QtGui

catalogue_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'icons_catalogue')

class Icon(QG.QIcon):
	def __init__(self,fileName,*args,**kwargs):
		path = os.path.join(catalogue_dir,fileName)
		super(Icon,self).__init__(path,*args,**kwargs)





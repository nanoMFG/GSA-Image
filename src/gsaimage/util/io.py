import json
import logging
import os
import subprocess
from collections import deque

import cv2
import requests
from PyQt5 import QtGui, QtCore, QtWidgets
from util.util import errorCheck

logger = logging.getLogger(__name__)

IMPORT_LOCATION = "/apps/importfile/bin/importfile"

class DownloadThread(QtCore.QThread):
    """
    Threading class for downloading files. Can be used to download files in parallel.

    url:                    Box download url.
    thread_id:              Thread ID that used to identify the thread.
    info:                   Dictionary for extra parameters.
    """

    downloadFinished = QtCore.pyqtSignal(object, int, object)

    def __init__(self, url, thread_id, info={}):
        super(DownloadThread, self).__init__()
        self.url = url
        self.thread_id = thread_id
        self.info = info
        self.data = None

        self.finished.connect(self.signal)

    def __del__(self):
        self.wait()

    def signal(self):
        self.downloadFinished.emit(self.data, self.thread_id, self.info)

    def run(self):
        r = requests.get(self.url)
        self.data = r.content

class DownloadRunner(QtCore.QObject):
    """
    Allows for download interruption by intercepting and preventing signal. Does not actually terminate thread.
    """
    finished = QtCore.pyqtSignal([],[object, int, object])
    terminated = QtCore.pyqtSignal()
    def __init__(self,thread,parent=None):
        super(DownloadRunner,self).__init__(parent=parent)
        self.thread = thread
        self.interrupted = False

        self.thread.downloadFinished.connect(self.sendSignal)

    def sendSignal(self,*args):
        if self.interrupted == False:
            self.finished[object, int, object].emit(*args)
            self.finished.emit()

    def interrupt(self):
        self.interrupted = True
        self.terminated.emit()

    def start(self):
        print('Runner [%s] started.'%self.thread.thread_id)
        self.thread.start()

class DownloadPool(QtCore.QObject):
    """
    Thread pooler for handling download threads

    max_thread_count:           (int) Max number of threads running at once.
    """
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    terminated = QtCore.pyqtSignal()
    def __init__(self,max_thread_count=1):
        super(DownloadPool,self).__init__()
        self.max_thread_count = max_thread_count
        self._count = 0
        self.queue = deque()
        self.running = []

        self.started.connect(lambda: print("Threads Started. Count: %s"%self.count()))

    def maxThreadCount(self):
        return self.max_thread_count

    def count(self):
        return self._count

    def queueCount(self):
        return len(self.queue)

    def runCount(self):
        return len(self.running)

    def doneCount(self):
        return self.count() - self.runCount() - self.queueCount()

    def addThread(self,thread):
        assert isinstance(thread,DownloadThread)
        # 'terminated' signal tells all runners to prevent DownloadThread 'finished' signal from emitting
        runner = DownloadRunner(thread)
        self.terminated.connect(runner.interrupt)
        # When runner is interrupted or its DownloadThread finishes, thread is removed from pool
        runner.finished.connect(lambda: self.running.remove(runner) if runner in self.running else None)
        runner.terminated.connect(lambda: self.queue.remove(runner) if runner in self.queue else None)
        # When thread finishes, run next thread in queue
        runner.finished.connect(self.runNext)

        self.queue.append(runner)
        self._count += 1

        print("Thread added: %s (Count: %s)"%(runner.thread.thread_id,self.count()))

        return runner

    def runNext(self):
        if len(self.queue) > 0 and len(self.running) <= self.max_thread_count:
            runner = self.queue.popleft()
            self.running.append(runner)
            runner.start()
        elif len(self.running) == 0:
            self.finished.emit()

    def run(self):
        self.started.emit()
        while len(self.running) < self.max_thread_count and len(self.queue) > 0:
            self.runNext()

    def terminate(self):
        print("Thread pool termminated. (Queue: %s, Running: %s)"%(self.queueCount(),self.runCount()))
        self.terminated.emit()
        self._count = 0



class IO(QtWidgets.QWidget):
    """
    Widget for handling import/export.

    config:             (object ConfigParams) holds configuration for IO mode.
    imptext:            (str, None) Text on import button. None removes import button.
    exptext:            (str, None) Text on export button. None removes export button.
    """
    importClicked = QtCore.pyqtSignal(object)
    exportClicked = QtCore.pyqtSignal()
    exportData = QtCore.pyqtSignal(object)
    def __init__(self,config,imptext="Import",exptext="Export",default_filename=None,ftype='json',extension=None,parent=None):
        super(IO,self).__init__(parent=parent)
        self.config = config

        self.layout = QtGui.QGridLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        if isinstance(imptext,str):
            self.import_button = QtWidgets.QPushButton(imptext)
            self.import_button.clicked.connect(self.importFile)
            self.layout.addWidget(self.import_button,0,0)
        if isinstance(exptext,str):
            self.export_button = QtWidgets.QPushButton(exptext)
            self.export_button.clicked.connect(lambda: self.exportClicked.emit())
            self.exportData.connect(
                lambda data: self.exportFile(
                    data=data,
                    default_filename=default_filename,
                    ftype=ftype,
                    extension=extension))
            self.layout.addWidget(self.export_button,0,1)

    def requestExportData(self):
        self.exportClicked.emit()

    # @errorCheck(error_text="Error importing file!")
    def importFile(self):
        if self.config.mode == "local":
            file_path = QtGui.QFileDialog.getOpenFileName()
            if isinstance(file_path, tuple):
                file_path = file_path[0]
            else:
                pass
            self.importClicked.emit(file_path) 
        elif self.config.mode == "nanohub":
            file_path = (
                subprocess.check_output(IMPORT_LOCATION, shell=True)
                .strip()
                .decode("utf-8")
            )
            self.importClicked.emit(file_path) 
        else:
            pass

    def saveLocal(self,data,filename,ftype=None):
        with open(filename,'w') as f:
            if ftype == None:
                f.write(filename)
            elif ftype == 'json':
                json.dump(data,f)
            elif ftype == 'image':
                cv2.imwrite(filename,data)
            else:
                raise ValueError("Parameter 'ftype' must be 'image', 'json' or 'None'!")
        return filename

    @errorCheck(error_text="Error exporting file!")
    def exportFile(self,data,default_filename=None,ftype='json',extension=None):
        if data is None:
            return
        if isinstance(extension,str) == False and extension is not None:
            raise ValueError("Parameter 'extension' must be of type 'str' or 'None'!")
        if ftype == 'image' and extension is None:
            raise ValueError("Parameter 'extension' cannot be 'None' if ftype is 'image'!")

        if ftype=='json':
            if extension is None:
                extension = 'json'
            elif extension != 'json' or extension != '.json':
                raise ValueError("Parameter 'ftype' is 'json' but parameter 'extension' is not 'None' or 'json'!")

        if isinstance(extension,str):
            extension = extension.strip('.').lower()
        if isinstance(default_filename,str):
            filename = default_filename
        else:
            filename = 'untitled'
        if isinstance(extension,str) and filename.split('.')[-1].lower() != extension:
            filename = filename + ".%s"%extension

        directory = os.path.join(os.getcwd(),filename)
        if self.config.mode == 'local':
            if isinstance(extension,str):
                filt = "%s (*.%s)"%(extension.upper(),extension.lower())
            else:
                filt = ''
            filename = QtWidgets.QFileDialog.getSaveFileName(None, 
                "Export Image", 
                directory,
                filt)[0]
            if filename != '':
                self.saveLocal(data,directory,ftype)
        elif self.config.mode == 'nanohub':
            self.saveLocal(data,directory,ftype)
            # print(filename, os.path.isfile(filename))
            subprocess.check_output('exportfile %s'%directory,shell=True)
            # os.remove(filename)


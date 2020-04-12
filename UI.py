import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from API import run_FCN,run_DeepLab,run_GCN

class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        self.img_path = None
        self.pix = QPixmap(256, 256)
        self.pix.fill(Qt.white)
        
    def initUI(self):         
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('Ready')

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('File')
        modelMenu = menubar.addMenu('Model')
        helpAction = QAction('Help',self)
        menubar.addAction(helpAction)
        helpAction.triggered.connect(self.help)
        
        openAct = QAction('Open',self)
        fileMenu.addAction(openAct)
        openAct.triggered.connect(self.load_img)

        

        fcnAct = QAction('FCN',self)
        modelMenu.addAction(fcnAct)
        fcnAct.triggered.connect(self.run_fcn)  

        deeplabAct = QAction('DeepLab',self)
        modelMenu.addAction(deeplabAct)
        deeplabAct.triggered.connect(self.run_deeplab)

        gcnAct = QAction('GCN',self)
        modelMenu.addAction(gcnAct)
        gcnAct.triggered.connect(self.run_gcn)

        self.label_img = QLabel(self)
        self.label_img.setFixedSize(256, 256)
        self.label_img.move(50, 30)
        self.label_img.setStyleSheet("QLabel{background:white;}")

        self.label_pred = QLabel(self)
        self.label_pred.setFixedSize(256,256)
        self.label_pred.move(350, 30)
        self.label_pred.setStyleSheet("QLabel{background:white;}")


        self.setGeometry(100,100,650,340)
        self.center()
        self.setWindowTitle('AerialSeg-GUI')    
        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def load_img(self):
        print("load_img")
        imgName, imgType = QFileDialog.getOpenFileName(self,caption="Choose an image",
                                                        directory="Desktop/UI/",
                                                        filter="Image Files(*.bmp *.png *.jpg);;All Files(*)")
        self.img_path = imgName
        img = QtGui.QPixmap(imgName).scaled(self.label_img.width(), self.label_img.height())
        self.label_img.setPixmap(img)
        self.label_pred.setPixmap(self.pix)

    def help(self):
        QMessageBox.information(self, "Help Information", "Load an image and then choose the algorithm you want. Left side is the loaded image while right side is the segmentation mask.",QMessageBox.Ok)

    def run_fcn(self):
        print("run_fcn")
        if self.img_path == None:
            QMessageBox.critical(self, "Error", "Please load an image first!")
            return
        else:
            mask = run_FCN(self.img_path)
            img = QtGui.QPixmap(mask).scaled(self.label_pred.width(), self.label_pred.height())
            self.label_pred.setPixmap(img)

    def run_deeplab(self):
        print("run_deeplab")
        if self.img_path == None:
            QMessageBox.critical(self, "Error", "Please load an image first!")
            return
        else:
            mask = run_DeepLab(self.img_path)
            img = QtGui.QPixmap(mask).scaled(self.label_pred.width(), self.label_pred.height())
            self.label_pred.setPixmap(img)

    def run_gcn(self):
        print("run_gcn")
        if self.img_path == None:
            QMessageBox.critical(self, "Error", "Please load an image first!")
            return
        else:
            pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
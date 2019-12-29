from imutils import face_utils
import imutils
import dlib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import sys
import cv2
import eos
import numpy as np

sys._excepthook = sys.excepthook


def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# Set the exception hook to our wrapping function
sys.excepthook = my_exception_hook


class Ui_MainWindow(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=None)
        self.fileName = None
        self.setObjectName("MainWindow")
        #        self.resize(503, 300)
        #        self.setGeometry(50, 50, 500, 300)
        self.setFixedSize(500, 300)
        self.setAutoFillBackground(False)
        self.fileselect = QtWidgets.QPushButton(self)
        self.fileselect.setGeometry(QtCore.QRect(20, 30, 261, 23))
        self.fileselect.setObjectName("fileselect")
        self.fileselect.clicked.connect(self.loadFile)

        self.graphicsView = QtWidgets.QLabel(self)
        self.graphicsView.setGeometry(QtCore.QRect(20, 70, 261, 201))
        self.graphicsView.setObjectName("graphicsView")

        self.viewsharpen = QtWidgets.QPushButton(self)
        self.viewsharpen.setGeometry(QtCore.QRect(400, 70, 81, 23))
        self.viewsharpen.setObjectName("viewsharpen")
        self.viewsharpen.clicked.connect(self.vsharp)
        self.viewsharpen.setEnabled(False)

        self.noiseview = QtWidgets.QPushButton(self)
        self.noiseview.setGeometry(QtCore.QRect(400, 130, 81, 23))
        self.noiseview.setObjectName("noiseview")
        self.noiseview.clicked.connect(self.vnoise)
        self.noiseview.setEnabled(False)

        self.eroview = QtWidgets.QPushButton(self)
        self.eroview.setGeometry(QtCore.QRect(400, 250, 81, 23))
        self.eroview.setObjectName("eroview")
        self.eroview.clicked.connect(self.vero)
        self.eroview.setEnabled(False)

        self.spaitialview = QtWidgets.QPushButton(self)
        self.spaitialview.setGeometry(QtCore.QRect(400, 190, 81, 23))
        self.spaitialview.setObjectName("spaitialview")
        self.spaitialview.clicked.connect(self.vspaitial)
        self.spaitialview.setEnabled(False)

        self.execute = QtWidgets.QPushButton(self)
        self.execute.setGeometry(QtCore.QRect(300, 30, 181, 23))
        self.execute.setObjectName("execute")
        self.execute.clicked.connect(self.e_xecute)
        self.execute.setEnabled(False)

        self.checksharp = QtWidgets.QCheckBox(self)
        self.checksharp.setGeometry(QtCore.QRect(300, 70, 91, 20))
        self.checksharp.setObjectName("checksharp")
        self.checknoise = QtWidgets.QCheckBox(self)
        self.checknoise.setGeometry(QtCore.QRect(300, 130, 81, 20))
        self.checknoise.setObjectName("checknoise")
        self.checkspaital = QtWidgets.QCheckBox(self)
        self.checkspaital.setGeometry(QtCore.QRect(300, 190, 70, 17))
        self.checkspaital.setObjectName("checkspaital")
        self.checkero = QtWidgets.QCheckBox(self)
        self.checkero.setGeometry(QtCore.QRect(300, 250, 70, 17))
        self.checkero.setObjectName("checkero")

        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "3D Morphological Model"))
        self.fileselect.setText(_translate("MainWindow", "File"))
        self.viewsharpen.setText(_translate("self", "View"))
        self.noiseview.setText(_translate("MainWindow", "View"))
        self.eroview.setText(_translate("MainWindow", "View"))
        self.spaitialview.setText(_translate("MainWindow", "View"))
        self.checksharp.setText(_translate("MainWindow", "Sharpen"))
        self.checknoise.setText(_translate("MainWindow", "Noise"))
        self.checkspaital.setText(_translate("MainWindow", "Spaitial"))
        self.checkero.setText(_translate("MainWindow", "Erosion"))
        self.execute.setText(_translate("MainWindow", "Generate"))

    def loadFile(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", "", "Image Files (*.jpg *.png)")
        self.noiseview.setEnabled(True)
        self.execute.setEnabled(True)
        self.eroview.setEnabled(True)
        self.noiseview.setEnabled(True)
        self.viewsharpen.setEnabled(True)
        self.spaitialview.setEnabled(True)
        pic = QtGui.QPixmap(self.fileName)
        try:
            pixmap = pic.scaled(261, 201)
            self.graphicsView.setPixmap(pixmap)
            self.graphicsView.show()
            self.graphicsView.image_loaded = True

        except Exception as e:
            print(str(e))

        return

    def vsharp(self):
        image = cv2.imread(self.fileName)
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        imS = cv2.resize(sharpened, (400, 400))
        cv2.imshow('Sharp', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def vnoise(self):
        image = cv2.imread(self.fileName)
        median = cv2.medianBlur(image, 5)
        imS = cv2.resize(image, (400, 400))
        cv2.imshow('Noise R', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def vero(self):
        image = cv2.imread(self.fileName)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(image, kernel, iterations=1)
        imS = cv2.resize(erosion, (400, 400))
        cv2.imshow('Ero', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def vspaitial(self):
        image = cv2.imread(self.fileName)
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        imS = cv2.resize(blur, (400, 400))
        cv2.imshow('Spaitial', imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def e_xecute(self):
        image = cv2.imread(self.fileName)

        if self.checksharp.isChecked():
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel_sharpening)
        if self.checknoise.isChecked():
            image = cv2.medianBlur(image, 5)
        if self.checkspaital.isChecked():
            kernel = np.ones((5, 5), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
        if self.checkero.isChecked():
            image = cv2.GaussianBlur(image, (5, 5), 0)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

        landmarks = []
        ibug_index = 1
        for s in shape:
            landmarks.append(eos.core.(str(ibug_index), [float(s[0]), float(s[1])]))
            ibug_index = ibug_index + 1

        image_width = image.shape[1]
        image_height = image.shape[0]
        model = eos.morphablemodel.load_model("eos/sfm_shape_3448.bin")
        blendshapes = eos.morphablemodel.load_blendshapes("eos/expression_blendshapes_3448.bin")
        morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                            color_model=eos.morphablemodel.PcaModel(),
                                                                            vertex_definitions=None,
                                                                            texture_coordinates=model.get_texture_coordinates())
        landmark_mapper = eos.core.LandmarkMapper('eos/ibug_to_sfm.txt')
        edge_topology = eos.morphablemodel.load_edge_topology('eos/sfm_3448_edge_topology.json')
        contour_landmarks = eos.fitting.ContourLandmarks.load('eos/ibug_to_sfm.txt')
        model_contour = eos.fitting.ModelContour.load('eos/sfm_model_contours.json')

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
                                                                                       landmarks, landmark_mapper,
                                                                                       image_width,
                                                                                       image_height, edge_topology,
                                                                                       contour_landmarks, model_contour)
        model_3d = eos.core.Mesh()
        model_3d = eos.morphablemodel.sample_to_mesh(model.get_shape_model().draw_sample(shape_coeffs),
                                                     model.get_color_model().get_mean(),
                                                     model.get_shape_model().get_triangle_list(),
                                                     model.get_color_model().get_triangle_list(),
                                                     model.get_texture_coordinates())
        eos.core.write_textured_obj(mesh=model_3d, filename='./3dmesh.obj')
        isomap = eos.render.extract_texture(mesh, pose, image)
        rows = isomap.shape[0]
        cols = isomap.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        isomap = cv2.warpAffine(isomap, M, (cols, rows))
        cv2.imwrite("./3dmesh.isomap.png", isomap)
        QMessageBox.about(self, "3dmesh", "Completed")

        return


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = Ui_MainWindow()
    w.show()
    sys.exit(app.exec_())

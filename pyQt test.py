import sys, traceback
from programma_henk import Model

from PyQt6.QtCore import Qt, QSize, QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QComboBox,
    QProgressBar
)
import pyqtgraph as pg
import numpy as np
from scipy.optimize import curve_fit
from collections import OrderedDict


pg.setConfigOptions(antialias=True)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress
    
    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setup window properties
        self.setWindowTitle("VEPFIT")
        self.resize(QSize(1200,800))

        # Layout setup
        main_layout = QGridLayout()
        param_layout = QGridLayout()
        graph_layout = QVBoxLayout()

        graph_layout.setContentsMargins(20,20,20,20)
        graph_layout.setSpacing(20)

        main_layout.addLayout(param_layout, 0,0, 1,1)
        main_layout.addLayout(graph_layout, 0,1, 1,1)
        main_layout.setColumnStretch(0,3)
        main_layout.setContentsMargins(20,20,20,20)

        # # The graphing window
        # self.graph = GraphWindow()

        # graphbt = QPushButton("Show Graph")
        # graphbt.clicked.connect(self.show_graph)

        # layout.addWidget(graphbt, 1,0)

        param_layout.addWidget(QLabel("<span style='font-size:20px;'>Program and experiment parameters</span>",), 0,0,1,2)

        exp_data_dropdown = QComboBox()
        exp_data_dropdown.addItems(["../exp data.csv", "../exp data E.csv"])
        exp_data_dropdown.currentTextChanged.connect(self.new_data_file)
        param_layout.addWidget(exp_data_dropdown, 0,2)

        program_layout = QGridLayout()
        param_layout.addLayout(program_layout, 1,0,1,3)

        self.parameters = OrderedDict()
        #                              Density,              L_p,             EV,                S,              W
        self.parameters["Surface"] = [None,              QDoubleSpinBox(), None,     QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["Al"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["Al/SiO"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["SiO"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["SiO/Si 1"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["SiO/Si 2"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        self.parameters["Si"] = [QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]


        program_layout.addWidget(QLabel("Density:"),1,0)
        program_layout.addWidget(QLabel("Diff. Length:"),2,0)
        program_layout.addWidget(QLabel("EV:"),3,0)
        program_layout.addWidget(QLabel("S:"),4,0)
        program_layout.addWidget(QLabel("W:"),5,0)

        index = 1
        for layer in self.parameters.keys():
            program_layout.addWidget(QLabel(layer), 0, index)
            row = 1
            for widget in self.parameters[layer]:
                if not widget:
                    row+=1
                    continue
                program_layout.addWidget(widget, row, index)
                widget.setMinimumWidth(100)
                

                if row in [4,5]:
                    widget.setSingleStep(0.001)
                    widget.setMinimum(-1)
                    widget.setMaximum(3)
                    widget.setDecimals(5)
                    widget.valueChanged.connect(self.program_param_changed)
                if row in [1,2,3]:
                    widget.valueChanged.connect(self.remodel_necessary)

                if row==1:
                    widget.setValue(2.33)
                    widget.setRange(0,10)
                if row==2:
                    widget.setRange(0,1000)
                if row==3:
                    widget.setRange(-100000,100000)
                
                row+=1

            index+=1


        model_button = QPushButton("Model")
        model_button.clicked.connect(self.start_model)
        param_layout.addWidget(model_button, 6,0, 1,2)

        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.fit)
        self.fit_win = None
        param_layout.addWidget(fit_button, 6,2)

        self.model_progress = QProgressBar()
        param_layout.addWidget(self.model_progress, 7,0,1,3)
        self.model_progress.hide()

        # Graphs
        self.exp_data_path = "../exp data.csv"

        self.graph1 = pg.PlotWidget()
        graph_layout.addWidget(self.graph1)
        self.graph1.setBackground("w")
        self.graph1.setLabel("left", "<span style='color:black;'>S</span>")
        self.graph1.setLabel("bottom", "<span style='color:black;'>E (keV)</span>")

        self.graph2 = pg.PlotWidget()
        graph_layout.addWidget(self.graph2)
        self.graph2.setBackground("w")
        self.graph2.setLabel("left", "<span style='color:black;'>W</span>")
        self.graph2.setLabel("bottom", "<span style='color:black;'>E (keV)</span>")

        self.graph3 = pg.PlotWidget()
        param_layout.addWidget(self.graph3, 8,0,3,3)
        self.graph3.setBackground("w")
        self.graph3.setLabel("left", "<span style='color:black;'>W</span>")
        self.graph3.setLabel("bottom", "<span style='color:black;'>S</span>")

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.toolbar = self.menuBar()
        # self.addToolBar(self.toolbar)

        save_params = QAction("Save parameters", self)
        save_params.triggered.connect(self.save_params)
        self.toolbar.addAction(save_params)

        load_params = QAction("Load parameters", self)
        load_params.triggered.connect(self.load_params)
        self.toolbar.addAction(load_params)

        # Load model
        self.modelled = False
        self.remodel = True
        self.model = Model()

        self.threadpool = QThreadPool()

    def new_data_file(self, fn):
        self.exp_data_path = fn
        self.program_param_changed()

    def save_params(self):
        export = np.zeros((len(self.parameters.keys()), 5))
        header = []
        i = 0
        for layer in self.parameters:
            header.append(layer)

            j = 0
            for widget in self.parameters[layer]:
                if not widget:
                    export[i,j] = 0
                    j+=1
                    continue

                export[i,j] = widget.value()
                j+=1

            i+=1
        # print(header, export)
        np.savetxt("parameters.csv", export, delimiter=",")

    def load_params(self):
        imp = np.loadtxt("parameters.csv", delimiter=",")

        i = 0
        for layer in self.parameters:
            j = 0
            for widget in self.parameters[layer]:
                if not widget:
                    j+=1
                    continue

                widget.setValue(imp[i,j])
                j+=1

            i+=1

    def remodel_necessary(self):
        self.remodel=True

    def update_model_params(self):
        self.model.rho = np.zeros(self.model.N+1)
        self.model.rho[self.model.Al_index] = self.parameters["Al"][0].value()
        self.model.rho[self.model.AlSio_index] = self.parameters["Al/SiO"][0].value()
        self.model.rho[self.model.Sio_index] = self.parameters["SiO"][0].value()
        self.model.rho[self.model.Siosi1] = self.parameters["SiO/Si 1"][0].value()
        self.model.rho[self.model.Siosi2] = self.parameters["SiO/Si 2"][0].value()
        self.model.rho[self.model.Si_index] = self.parameters["Si"][0].value()
        self.model.rho[-1] = self.parameters["Si"][0].value()

        self.model.L_p = np.zeros(self.model.N+1) # nm
        self.model.L_p[self.model.Al_index] = self.parameters["Al"][1].value()
        self.model.L_p[self.model.AlSio_index] = self.parameters["Al/SiO"][1].value()
        self.model.L_p[self.model.Sio_index] = self.parameters["SiO"][1].value()
        self.model.L_p[self.model.Siosi1] = self.parameters["SiO/Si 1"][1].value()
        self.model.L_p[self.model.Siosi2] = self.parameters["SiO/Si 2"][1].value()
        self.model.L_p[self.model.Si_index] = self.parameters["Si"][1].value()
        self.model.L_p[-1] = self.parameters["Si"][1].value()

        self.model.EV = np.zeros(self.model.N+1) # nm
        self.model.EV[self.model.Al_index] = self.parameters["Al"][2].value()
        self.model.EV[self.model.AlSio_index] = self.parameters["Al/SiO"][2].value()
        self.model.EV[self.model.Sio_index] = self.parameters["SiO"][2].value()
        self.model.EV[self.model.Siosi1] = self.parameters["SiO/Si 1"][2].value()
        self.model.EV[self.model.Siosi2] = self.parameters["SiO/Si 2"][2].value()
        self.model.EV[self.model.Si_index] = self.parameters["Si"][2].value()
        self.model.EV[-1] = 0


        self.model.S_i = np.zeros(self.model.N)
        self.model.S_surf = self.parameters["Surface"][3].value()
        self.model.S_i[self.model.Al_index] = self.parameters["Al"][3].value()
        self.model.S_i[self.model.AlSio_index] = self.parameters["Al/SiO"][3].value()
        self.model.S_i[self.model.Sio_index] = self.parameters["SiO"][3].value()
        self.model.S_i[self.model.Siosi1] = self.parameters["SiO/Si 1"][3].value()
        self.model.S_i[self.model.Siosi2] = self.parameters["SiO/Si 2"][3].value()
        self.model.S_i[self.model.Si_index] = self.parameters["Si"][3].value()

        self.model.W_i = np.zeros(self.model.N)
        self.model.W_surf = self.parameters["Surface"][4].value()
        self.model.W_i[self.model.Al_index] = self.parameters["Al"][4].value()
        self.model.W_i[self.model.AlSio_index] = self.parameters["Al/SiO"][4].value()
        self.model.W_i[self.model.Sio_index] = self.parameters["SiO"][4].value()
        self.model.W_i[self.model.Siosi1] = self.parameters["SiO/Si 1"][4].value()
        self.model.W_i[self.model.Siosi2] = self.parameters["SiO/Si 2"][4].value()
        self.model.W_i[self.model.Si_index] = self.parameters["Si"][4].value()

        self.model.update_val()
    
    def program_param_changed(self):
        if not self.remodel:
            self.start_model()

    def start_model(self):
        self.update_model_params()

        if self.remodel:
            self.remodel = False
            self.model_progress.show()
            self.model_progress.setMaximum(self.model.N)

            worker = Worker(self.model.T_ij)
            worker.signals.progress.connect(self.update_progress)
            worker.signals.finished.connect(self.finish_model)

            self.threadpool.start(worker)
        else:
            self.finish_model()

    def update_progress(self, n):
        self.model_progress.setValue(n)

    def finish_model(self):
        self.model_progress.hide()
        self.model_progress.setValue(0)

        E = self.model.E_space
        S_e = np.zeros(len(E))
        W_e = np.zeros(len(E))

        exp_data = np.loadtxt(self.exp_data_path, delimiter=",", skiprows=1)
        # print(exp_data.shape)
        # print(exp_data)
        E_data = exp_data[:,0]
        W_data = exp_data[:,1]
        S_data = exp_data[:,2]


        for i in range(len(S_e)):
            S_e[i] = self.model.S(E[i])
            W_e[i] = self.model.W(E[i])

        if not self.modelled:
            self.modelled = True
            pen = pg.mkPen(color="#32a852", width = 3)
            self.model_S = self.graph1.plot(E, S_e,pen=pen)
            self.exp_S = self.graph1.plot(E_data,S_data, pen=pg.mkPen(color="#000000", width = 3))
            self.model_W = self.graph2.plot(E, W_e,pen=pen)
            self.exp_W = self.graph2.plot(E_data,W_data, pen=pg.mkPen(color="#000000", width = 3))
            self.model_SW = self.graph3.plot(S_e, W_e,pen=pen)
            self.exp_SW = self.graph3.plot(S_data,W_data, pen=pg.mkPen(color="#000000", width = 3))
        else:
            self.model_S.setData(E,S_e)
            self.model_W.setData(E, W_e)
            self.model_SW.setData(S_e,W_e)

            self.exp_S.setData(E_data,S_data)
            self.exp_W.setData(E_data, W_data)
            self.exp_SW.setData(S_data,W_data)

    def fit(self):
        print(self.fit_win)
        self.fit_win = FitWindow()
        self.fit_win.show()
        # Fit S and W parameters
        # exp_data = np.loadtxt(self.exp_data_path, delimiter=",", skiprows=1)
        # # print(exp_data)
        # E_data = exp_data[:,0]
        # W_data = exp_data[:,1]
        # S_data = exp_data[:,2]


        # poptS,pcovS = curve_fit(self.model.S_fit, E_data, S_data, bounds=(0.3,0.7), p0=[0.61, 0.6,0.556,0.5499,0.576,0.57], sigma=1/S_data)
        # poptW,pcovW = curve_fit(self.model.W_fit, E_data, W_data, bounds=(0,0.1), p0=[0.025,0.044, 0.054, 0.027, 0.027, 0.032])
        # self.start_model()

class FitWindow(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("This doesn't work yet"))
        self.setLayout(layout)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
from tkinter import ttk, messagebox, IntVar
import tkinter as tk
from threading import Thread
import time

# from RWMBackend import RWMBackend

from .ProbabilityVectorCalculator import ProbabilityVectorCalculator
from .ArrivalTimesFormatter import ArrivalTimesFormatter
from .ModelFitter import ModelFitter
from .ModelEvaluatorMenu import EvaluatorMenu
# from ProbabilityVectorCalculator import ProbabilityVectorCalculator
# from ArrivalTimesFormatter import ArrivalTimesFormatter
# from ModelFitter import ModelFitter
# from ModelEvaluatorMenu import EvaluatorMenu

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import datetime
import dateutil
from copy import deepcopy
import os
import shutil
import pickle

from configparser import ConfigParser, ExtendedInterpolation

from ..MenuNode import MenuNode


class RandomWalkModellerMenu():

    # Creates a ConfigParser object which contains the default entries for the menu
    # Initializes a ProbabilityVectorCalculator object, which takes in an arrival times DF and outputs a set of probability vectors using a set of conditions
    # Initializes a RandomWalkModeller object, which takes the probability vectors as input and fits a transition matrix to them
    def __init__(self, lineDataDir, menu=None):
        self.pvc = ProbabilityVectorCalculator()
        self.mf = ModelFitter()
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('configs/default.ini')
        if type(menu) != type(None):
            self.menu = menu
            self.mainWindow = self.menu.window

        self.lineDataDir = lineDataDir
        if os.path.exists(os.path.join(self.lineDataDir, 'model_data')) == False:
            os.mkdir(os.path.join(self.lineDataDir, 'model_data'))


    # Creates menu for user to select options for Random Walk Modeller
    def initialize(self):

        self.borderx = 10
        self.gapx = 5
        self.gapy = 5

        #---------------------------- Initialization ----------------------------------------
        self.mainWindow.title("RandomWalkModeller")

        self.welcomeLabel = tk.Label(text="Random Walk Modeller", master=self.mainWindow)
        self.welcomeLabel.pack(padx=10, pady=5)
        #------------------------------------------------------------------------------------------------------------------------------------

        #------------------------------------ Current line data folder ----------------------------------------------------------
        self.lineDataFolderLabel = tk.Label(text="Current line data folder: "+os.path.split(self.lineDataDir)[1], master=self.mainWindow)
        self.lineDataFolderLabel.pack(side="top", padx=self.borderx, pady=self.gapy)
        #----------------------------------------------------------------------------------------------------------------------

        #---------------------------------- Model directory/Prob. vector creation --------------------------------------------------
        # CREATED MODEL DIRECTORY CONTAINS A FIXED SET OF PROBABILITY VECTORS
        # THESE PROB VECTORS ARE CALCULATED ACCORDING TO SOME USER-GIVEN PARAMETERS, SUCH AS TIME STEP VALUE AND DEPARTURE TIME RANGE
            # SINCE THE VECTORS ARE FIXED, THESE PARAMETERS ARE ALSO CONSTANT FOR A MODEL DIRECTORY
        # SETS OF PROB. VECTORS ARE GROUPED ACCORDING TO THE INITIAL LOCATIONS THE BUSES ARE OBSERVED AT
        # THE PROB. VECTORS ARE THEN USED IN THE COMPUTATION OF RANDOM WALK TRANSITION MATRICES IN THE NEXT SECTION

        self.modelDirCreationFrame = tk.LabelFrame(text="Model Directory Creation", master=self.mainWindow)
        self.modelDirCreationFrame.pack(side="top", fill='x', padx=self.borderx, pady=self.gapy)

        self.copyTrainJourneysLabel = tk.Label(text="Use same train journeys as", master=self.modelDirCreationFrame)
        self.copyTrainJourneysLabel.grid(column=0, row=0, sticky='E')
        self.copyTrainJourneysVar = tk.StringVar()
        self.copyTrainJourneysVar.trace(mode='w', callback=self.toggleJourneyCriteriaState)
        self.copyTrainJourneysEntry = ttk.Combobox(values=['None']+os.listdir(os.path.join(self.lineDataDir, 'model_data')), textvariable=self.copyTrainJourneysVar, state='readonly', master=self.modelDirCreationFrame)
        self.copyTrainJourneysEntry.grid(column=1, row=0, sticky='W')

        #----------------------- Journey criteria -------------------------------------------
        self.journeyCriteriaFrame = tk.LabelFrame(text="Journey inclusion criteria", master=self.modelDirCreationFrame)
        self.journeyCriteriaFrame.grid(column=0, columnspan=2, row=1, rowspan=3)
        # Route selection
        self.routeSelFrame = tk.Frame(master=self.journeyCriteriaFrame)
        self.routeSelFrame.grid(column=0, columnspan=2, row=0)
        self.routeSelLabel = tk.Label(text="Route", master=self.routeSelFrame)
        # self.routeSelLabel.grid(column=0, row=0, sticky='E')
        self.routeSelLabel.pack(side="left")
        self.routeSelVar = tk.StringVar()
        self.routeSelVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        # self.routeSelVar.set(self.config['RandomWalkModeller']['route'])
        self.routeSelection = ttk.Combobox(master=self.routeSelFrame, textvariable=self.routeSelVar, values=os.listdir(os.path.join(self.lineDataDir, 'arrivals')), width=20, state='readonly')
        self.routeSelection.pack(side="left")
        # Displays total number of recorded journeys for the selected route
            # Note: These journeys are not necessarily complete
        self.routeNumJourneysLabel = tk.Label(master=self.modelDirCreationFrame)#self.routeNumJourneysLabel = tk.Label(master=self.modelDirCreationFrame)
        self.routeNumJourneysLabel.grid(column=2, row=1, sticky='W')#self.routeNumJourneysLabel.pack(side="left")
        # Departure time bounds
        self.depTimeFrame = tk.LabelFrame(text="Departure time range", master=self.journeyCriteriaFrame)
        self.depTimeFrame.grid(column=0, columnspan=2, row=1)
        # Departure time lower bound
        self.depTimeLowLabel = tk.Label(text="Minimum", master=self.depTimeFrame)
        self.depTimeLowLabel.grid(column=0, row=0, sticky='E')
        self.depTimeLowHourFrame = tk.Frame(master=self.depTimeFrame)
        self.depTimeLowHourFrame.grid(column=1, row=0, sticky='W')
        self.depTimeLowHourLabel = tk.Label(text="Hour", master=self.depTimeLowHourFrame)
        self.depTimeLowHourLabel.pack(side="left")
        self.depTimeLowHourVar = tk.StringVar()
        self.depTimeLowHourVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        self.depTimeLowHourEntry = tk.Spinbox(from_=0, to_=23, increment=1, wrap=True, width=4, textvariable=self.depTimeLowHourVar, master=self.depTimeLowHourFrame) #, command=self.updateNumJourneysLabels)
        self.depTimeLowHourEntry.pack(side="left")
        self.depTimeLowMinFrame = tk.Frame(master=self.depTimeFrame)
        self.depTimeLowMinFrame.grid(column=2, row=0, sticky='W')
        self.depTimeLowMinLabel = tk.Label(text="Min", master=self.depTimeLowMinFrame)
        self.depTimeLowMinLabel.pack(side="left")
        self.depTimeLowMinVar = tk.StringVar()
        self.depTimeLowMinVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        self.depTimeLowMinEntry = tk.Spinbox(from_=0, to_=59, increment=1, wrap=True, width=4, textvariable=self.depTimeLowMinVar, master=self.depTimeLowMinFrame) #, command=self.updateNumJourneysLabels)
        self.depTimeLowMinEntry.pack(side="left")
        # Departure time upper bound
        self.depTimeHighLabel = tk.Label(text="Maximum", master=self.depTimeFrame)
        self.depTimeHighLabel.grid(column=0, row=1, sticky='E')
        self.depTimeHighHourFrame = tk.Frame(master=self.depTimeFrame)
        self.depTimeHighHourFrame.grid(column=1, row=1, sticky='W')
        self.depTimeHighHourLabel = tk.Label(text="Hour", master=self.depTimeHighHourFrame)
        self.depTimeHighHourLabel.pack(side="left")
        self.depTimeHighHourVar = tk.StringVar()
        self.depTimeHighHourVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        self.depTimeHighHourEntry = tk.Spinbox(from_=0, to_=23, increment=1, wrap=True, width=4, textvariable=self.depTimeHighHourVar, master=self.depTimeHighHourFrame) #, command=self.updateNumJourneysLabels)
        self.depTimeHighHourEntry.pack(side="left")
        self.depTimeHighMinFrame = tk.Frame(master=self.depTimeFrame)
        self.depTimeHighMinFrame.grid(column=2, row=1, sticky='W')
        self.depTimeHighMinLabel = tk.Label(text="Min", master=self.depTimeHighMinFrame)
        self.depTimeHighMinLabel.pack(side="left")
        self.depTimeHighMinVar = tk.StringVar()
        self.depTimeHighMinVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        self.depTimeHighMinEntry = tk.Spinbox(from_=0, to_=59, increment=1, wrap=True, width=4, textvariable=self.depTimeHighMinVar, master=self.depTimeHighMinFrame) #, command=self.updateNumJourneysLabels)
        self.depTimeHighMinEntry.pack(side="left")
        # Label to display how many recorded journeys are within the departure time range
        self.depTimeNumJourneysLabel = tk.Label(master=self.modelDirCreationFrame)
        self.depTimeNumJourneysLabel.grid(column=2, row=2, sticky='W')
        # self.depTimeNumJourneysLabel = tk.Label(master=self.numJourneysFrame)
        # self.depTimeNumJourneysLabel.grid(column=0, row=1, sticky='W')

        # Include journeys which have unknown departure time but arrive at some location in the range between the latest and earliest 
            # arrival at the location out of journeys with known departure time in the given range of departure times
        # Give the number of journeys next to it
        self.unkDepTimeLabel = tk.Label(text="Include late start journeys", master=self.journeyCriteriaFrame)
        self.unkDepTimeLabel.grid(column=0, row=2, sticky='E')
        self.unkDepTimeVar = tk.BooleanVar()
        # self.unkDepTimeVar.trace(mode='w', callback=self.updateNumJourneysLabels)
        # self.unkDepTimeVar.trace(mode='w')
        self.unkDepTimeEntry = tk.Checkbutton(variable=self.unkDepTimeVar, master=self.journeyCriteriaFrame, command=self.updateNumJourneysLabels)
        self.unkDepTimeEntry.grid(column=1, row=2, sticky='W')
        self.unkDepNumJourneysLabel = tk.Label(master=self.modelDirCreationFrame)
        self.unkDepNumJourneysLabel.grid(column=2, row=3, sticky='W')
        # self.unkDepNumJourneysLabel = tk.Label(master=self.numJourneysFrame)
        # self.unkDepNumJourneysLabel.grid(column=0, row=2, sticky='W')
        #----------------------------------------------------------------------------------------------------------------------------

        # Train/Test split
        # Defines what proportion of the journeys that satisfy the above user-given parameters are used for fitting the model
        # Thus, if, for example, the model is to be evaluated exclusively using journeys outside the dep. time range, train can be set to 100%
        self.trainTestLabel = tk.Label(text="Train/Test journey split (%)", master=self.modelDirCreationFrame)
        self.trainTestLabel.grid(column=0, row=5, sticky='E')
        self.trainTestEntryFrame = tk.Frame(master=self.modelDirCreationFrame)
        self.trainTestEntryFrame.grid(column=1, row=5, sticky='W')
        self.trainFrame = tk.Frame(master=self.trainTestEntryFrame)#self.trainFrame = tk.Frame(master=self.modelDirCreationFrame)
        self.trainFrame.pack(side="left")#self.trainFrame.grid(column=1, row=5, sticky='W')
        self.trainLabel = tk.Label(text="Train", master=self.trainFrame)
        self.trainLabel.pack(side="left")
        self.trainIntVar = tk.StringVar()
        self.trainEntry = tk.Spinbox(from_=0, to=100, width=3, textvariable=self.trainIntVar, master=self.trainFrame)
        self.trainEntry.pack(side="left")
        self.testFrame = tk.Frame(master=self.trainTestEntryFrame)#self.testFrame = tk.Frame(master=self.modelDirCreationFrame)
        self.testFrame.pack(side="left")#self.testFrame.grid(column=2, row=5, sticky='W')
        self.testLabel = tk.Label(text="Test", master=self.testFrame)
        self.testLabel.pack(side="left")
        self.testIntVar = tk.StringVar()
        self.testEntry = tk.Spinbox(from_=0, to=100, width=3, textvariable=self.testIntVar, master=self.testFrame)
        self.testEntry.pack(side="left")
        self.trainIntVar.trace(mode='w', callback=self.updateTestVal)
        self.testIntVar.trace(mode='w', callback=self.updateTrainVal)
        self.trainIntVar.set(self.config['RandomWalkModeller']['trainsplit'])

        # Defines the time gap between each computed prob. vector
        self.timeStepLabel = tk.Label(text="Time step (seconds)", master=self.modelDirCreationFrame)
        self.timeStepLabel.grid(column=0, row=6, sticky='E')
        self.timeStepEntry = tk.Spinbox(from_=0, to=999, width=3, increment=1, master=self.modelDirCreationFrame)
        self.timeStepEntry.grid(column=1, row=6, sticky='W')
        self.timeStepEntry.delete(0, 'end')
        self.timeStepEntry.insert(index=0, s=self.config['RandomWalkModeller']['timestepinterval'])

        self.stopsOnlyLabel = tk.Label(text="Stops Only", master=self.modelDirCreationFrame)
        self.stopsOnlyLabel.grid(column=0, row=7, sticky='E')
        self.stopsOnlyBool = tk.BooleanVar()
        self.stopsOnlyBool.set(self.config['RandomWalkModeller']['stopsonly'])
        self.stopsOnlyEntry = tk.Checkbutton(master=self.modelDirCreationFrame, variable=self.stopsOnlyBool)
        self.stopsOnlyEntry.grid(column=1, row=7, sticky='W')

        self.modelNameLabel = tk.Label(text="Model Name", master=self.modelDirCreationFrame)
        self.modelNameLabel.grid(column=0, row=8, sticky='E')
        self.modelNameEntry = tk.Entry(master=self.modelDirCreationFrame)
        self.modelNameEntry.grid(column=1, columnspan=2, row=8, sticky='W')
        self.modelNameEntry.insert(index=0, string=self.config['RandomWalkModeller']['vectorsetname'])

        self.lossFuncLabel = tk.Label(text="Loss Function", master=self.modelDirCreationFrame)
        self.lossFuncLabel.grid(column=0, row=9, sticky='E')
        self.lossFuncEntry = ttk.Combobox(values=['Wasserstein Distance', 'Kullback-Leibler Divergence', 'Arrival time error'], width=25, state='readonly', master=self.modelDirCreationFrame)
        self.lossFuncEntry.grid(column=1, columnspan=2, row=9, sticky='W')

        self.createModelDirButton = tk.Button(text="Create Model Directory", command=self.createModelDir, master=self.modelDirCreationFrame)
        self.createModelDirButton.grid(column=0, row=10, columnspan=3)
        #----------------------------------------------------------------------------------------------------------------------




        ############################ Model Directory Options ##########################################################
        # USER SELECTS A MODEL DIRECTORY, THEN BEGINS FITTING TRANSITION MATRIX PARAMETERS TO BEST FIT THE PROB VECTORS IN THE DIRECTORY
        # USER HAS THE OPTION TO STORE ALL THE COMPUTED PARAMETERS, OR ONLY THE MODEL WITH THE LOWEST ERROR
        self.modelOptionsFrame = tk.LabelFrame(text="Model Directory Options", master=self.mainWindow)
        self.modelOptionsFrame.pack(side="top", fill='x', padx=self.borderx)

        self.modelDirSelFrame = tk.Frame(master=self.modelOptionsFrame)
        self.modelDirSelFrame.pack(side="top", fill='x')
        self.modelDirSelLabel = tk.Label(text="Model Directory", master=self.modelDirSelFrame)
        self.modelDirSelLabel.grid(column=0, row=0, sticky='E')
        self.modelDirSelVar = tk.StringVar()
        self.modelDirSelVar.trace(mode='w', callback=self.updateBestLoss)
        self.modelDirSelEntry = ttk.Combobox(master=self.modelDirSelFrame, textvariable=self.modelDirSelVar, width=50, state='readonly', values=os.listdir(os.path.join(self.lineDataDir, 'model_data')))
        self.modelDirSelEntry.grid(column=1, columnspan=2, row=0, sticky='W')
        self.numModelsLabel = tk.Label(text="Number models displayed here", master=self.modelDirSelFrame)
        self.numModelsLabel.grid(column=0, columnspan=2, row=1)
        self.modelDirLossFuncLabel = tk.Label(text="Loss function given here", master=self.modelDirSelFrame)
        self.modelDirLossFuncLabel.grid(column=0, columnspan=2, row=2)
        self.bestLossLabel = tk.Label(text="Loss of best model displayed here", master=self.modelDirSelFrame)
        self.bestLossLabel.grid(column=0, columnspan=2, row=3)

        #---------------- Plot train vectors -----------------------------------------
        self.plotVectorsFrame = tk.LabelFrame(text="Train Vector Plotting", master=self.modelOptionsFrame)
        self.plotVectorsFrame.pack(side="top", fill='x', padx=self.borderx)

        self.minNumJourneysFrame = tk.Frame(master=self.plotVectorsFrame)
        self.minNumJourneysFrame.pack(side="top")
        self.minNumJourneysLabel = tk.Label(text="Min. Num Journeys", master=self.minNumJourneysFrame)
        self.minNumJourneysLabel.pack(side="left")
        self.minNumJourneysEntry = tk.Spinbox(from_=0, to_=99999, width=4, master=self.minNumJourneysFrame)
        self.minNumJourneysEntry.pack(side="left")

        self.plotVectorsButtonsFrame = tk.Frame(master=self.plotVectorsFrame)
        self.plotVectorsButtonsFrame.pack(side="top")
        self.plotTrainVectorsButton = tk.Button(text="Plot Train Vectors", command=self.plotTrainVectors, master=self.plotVectorsButtonsFrame)
        self.plotTrainVectorsButton.pack(side="left")
        self.stopPlottingVectorsButton = tk.Button(text="Stop Plotting", command=self.stopPlotting, master=self.plotVectorsButtonsFrame)
        self.stopPlottingVectorsButton.pack(side="left")
        #--------------------------------------------------------------------------

        #------------------ Plot train journeys ------------------------------------
        # self.plotJourneysFrame = tk.LabelFrame(text="Train Journey Plotting", master=self.modelOptionsFrame)
        # self.plotJourneysFrame.pack(side="top", fill='x', padx=self.borderx)

        # self.minNumJourneysFrame = tk.Frame(master=self.plotJourneysFrame)
        # self.minNumJourneysFrame.pack(side="top")
        # self.minNumJourneysLabel = tk.Label(text="Min. Num Journeys", master=self.minNumJourneysFrame)
        # self.minNumJourneysLabel.pack(side="left")
        # self.minNumJourneysEntry = tk.Spinbox(from_=0, to_=99999, width=4, master=self.minNumJourneysFrame)
        # self.minNumJourneysEntry.pack(side="left")
        # self.plotNormJourneysVar = tk.BooleanVar()
        # self.plotNormJourneysFrame = tk.Frame(master=self.plotJourneysFrame)
        # self.plotNormJourneysFrame.pack(side="top")
        # self.plotNormJourneysLabel = tk.Label(text="Normalize journeys", master=self.plotNormJourneysFrame)
        # self.plotNormJourneysLabel.pack(side="left")
        # self.plotNormJourneysCheck = tk.Checkbutton(variable=self.plotNormJourneysVar, master=self.plotNormJourneysFrame)
        # self.plotNormJourneysCheck.pack(side="left")

        self.plotTrainJourneysButton = tk.Button(text="Plot Normalized Train Journey Arrival Times", command=self.plotTrainJourneys, master=self.modelOptionsFrame)
        self.plotTrainJourneysButton.pack(side="top")
        #------------------------------------------------------------------------

        #---------------- Model fitting ------------------------
        self.modelFittingFrame = tk.LabelFrame(text="Model Fitting", master=self.modelOptionsFrame)
        self.modelFittingFrame.pack(side="top", fill='x', padx=self.borderx)

        self.maxNumIterationsFrame = tk.Frame(master=self.modelFittingFrame)
        self.maxNumIterationsFrame.pack(side="top")
        self.maxNumIterationsLabel = tk.Label(text="Max. num. Nelder-Mead iterations", master=self.maxNumIterationsFrame)
        self.maxNumIterationsLabel.pack(side="left")
        self.maxNumIterationsEntry = tk.Spinbox(from_=0, to_=99999999, width=6, master=self.maxNumIterationsFrame)
        self.maxNumIterationsEntry.delete(0, tk.END)
        self.maxNumIterationsEntry.insert(index=0, s='20000')
        self.maxNumIterationsEntry.pack(side="left")

        self.storeAllResultsFrame = tk.Frame(master=self.modelFittingFrame)
        self.storeAllResultsFrame.pack(side="top")
        self.storeAllResultsLabel = tk.Label(text="Store all results", master=self.storeAllResultsFrame)
        self.storeAllResultsLabel.pack(side="left")
        self.storeAllResultsBool = tk.BooleanVar()
        self.storeAllResultsButton = tk.Checkbutton(master=self.storeAllResultsFrame, variable=self.storeAllResultsBool)
        self.storeAllResultsButton.pack(side="left")

        self.modelFitButtonsFrame = tk.Frame(master=self.modelFittingFrame)
        self.modelFitButtonsFrame.pack(side="top")
        self.fitModelButton = tk.Button(text="Begin Model Fitting", command=self.fitModel, master=self.modelFitButtonsFrame)
        self.fitModelButton.pack(side="left")
        self.stopFittingButton = tk.Button(text="Stop Fitting", command=self.stopFitting, master=self.modelFitButtonsFrame)
        self.stopFittingButton.pack(side="left")
        self.fittingProgressLabel = tk.Label(text="Current model loss displayed here\nNumber of minimize iterations displayed here", master=self.modelFittingFrame)
        self.fittingProgressLabel.pack(side="top")
        #--------------------------------------------------------------------------------------------------


        #------------ Model Evaluation button --------------------------
        self.evalModelButton = tk.Button(text="Evaluate Models", command=self.beginEvaluation, master=self.modelOptionsFrame)
        self.evalModelButton.pack(side="top")
        # self.evalModelButton = tk.Button(text="Evaluate Models", command=self.beginEvaluation, master=self.modelDirSelFrame)
        # self.evalModelButton.grid(column=0, row=1)
        #----------------------------------------------------------------

        ##################################################################################################################

        self.mainWindow.mainloop()


    ######################################## Menu update functions ##########################################################
    def updateNumJourneysLabels(self, *args):
        try:
            self.pvc.continueGatheringJourneys = False

            # Retrieve DF to report the journey numbers
            if self.routeSelVar.get() == '':
                return
            
            arrivalsDF = pd.read_parquet(os.path.join(self.lineDataDir, 'arrivals', self.routeSelVar.get()))
            if 'JoureyCode' in arrivalsDF:
                arrivalsDF = arrivalsDF.drop('JourneyCode', axis=1)
            if 'VehicleRef' in arrivalsDF:
                arrivalsDF = arrivalsDF.drop('VehicleRef', axis=1)

            # Update overall number of journeys in route
            self.routeNumJourneysLabel['text'] = str(arrivalsDF.shape[0]) + ' journeys'

            # Update number of journeys between the departure times
            try:
                lowDepTime = datetime.time(hour=int(self.depTimeLowHourVar.get()), minute=int(self.depTimeLowMinVar.get()))
                highDepTime = datetime.time(hour=int(self.depTimeHighHourVar.get()), minute=int(self.depTimeHighMinVar.get()))
            except ValueError:
                self.depTimeNumJourneysLabel['text'] = 'Values should be int'
                self.unkDepNumJourneysLabel['text'] = 'Invalid departure time'
                return
            journeysInDepRange = self.pvc.gatherRelevantJourneys(arrivalTimes=arrivalsDF, departureTimeRange=(lowDepTime, highDepTime), addIntersectingJourneys=False)
            self.depTimeNumJourneysLabel['text'] = str(journeysInDepRange.shape[0]) + ' journeys'
            
            # Update number of journeys added through including journeys which have arrival time in range but have no known departure time
            # Get latest and earliest arrival time across all journeys for each location
            if self.unkDepTimeVar.get() == True:
                t = Thread(target=lambda:self.computeIntersectingJourneys(journeysInDepRange=journeysInDepRange))
                t.start()
                self.unkDepNumJourneysLabel['text'] = '...'
            else:
                self.unkDepNumJourneysLabel['text'] = self.depTimeNumJourneysLabel.get()
            return

            if self.unkDepTimeVar.get() == True and journeysInDepRange.shape[0] != 0:
                if journeysInDepRange.shape[0] == 0:
                    self.unkDepNumJourneysLabel['text'] = '0'
                    return
                unkDepTimes = arrivalsDF.loc[~arrivalsDF.index.isin(knownDepTimes.index)]
                self.getNumIntJourneys = Thread(target=lambda:self.computeIntersectingJourneys(unkDepTimes=unkDepTimes, journeysInDepRange=journeysInDepRange), daemon=True)
                self.getNumIntJourneys.start()
                self.unkDepNumJourneysLabel['text'] = '...'
                return
            elif self.unkDepTimeVar.get() == False:
                self.unkDepNumJourneysLabel['text'] = self.depTimeNumJourneysLabel['text']
        
        # As the window is being built, this function gets called before some of the variables referenced in it are initialized
        # This stops error being thrown because of these initial runs
        except AttributeError:
            return
        
    def computeIntersectingJourneys(self, journeysInDepRange):
        journeys = self.pvc.gatherRelevantJourneys(addIntersectingJourneys=True, arrivalTimes=journeysInDepRange, departureTimeRange=(datetime.time.min, datetime.time.max))
        if type(journeys) == type(None):
            return
        else:
            self.unkDepNumJourneysLabel['text'] = str(journeys.shape[0]) + ' journeys'
            return
        
        self.stopGettingAddedJourneys = False
        addedJourneys = pd.DataFrame(columns=unkDepTimes.columns)
        for c in journeysInDepRange:

            if self.stopGettingAddedJourneys == True:
                return

            if journeysInDepRange[c].dropna().shape[0] != 0:
                earliestArrival = min(journeysInDepRange[c].dropna().dt.time)
                latestArrival = max(journeysInDepRange[c].dropna().dt.time)

            # Add journeys to addedJourneys and remove them from unkDepTimes if their first known arrival time is in the range
            for j in unkDepTimes.index:
                if pd.isna(unkDepTimes.loc[j, c]) == False and (unkDepTimes.loc[j, c].time() <= latestArrival) and (unkDepTimes.loc[j, c].time() >= earliestArrival) and unkDepTimes.loc[j, :c].dropna().shape[0]==1:
                    addedJourneys.loc[j] = unkDepTimes.loc[j]
                    unkDepTimes = unkDepTimes.drop(index=j)
                
            # Journeys are kept in addedJourneys iff all their arrival times are in the arrival range for each location
            for j in addedJourneys.index:
                if pd.isna(addedJourneys.loc[j, c])==False and ((addedJourneys.loc[j, c].time() > latestArrival) or (addedJourneys.loc[j, c].time() < earliestArrival)):
                    addedJourneys = addedJourneys.drop(index=j)

        self.unkDepNumJourneysLabel['text'] = str(journeysInDepRange.shape[0] + addedJourneys.shape[0]) + ' journeys'

    def toggleJourneyCriteriaState(self, *args):
        copyModelDir = self.copyTrainJourneysVar.get()
        if copyModelDir == '' or copyModelDir == 'None':
            state = 'normal'
        else:
            state = 'disabled'
        self.iterativelySetFrameState(frame=self.journeyCriteriaFrame, state=state)
        self.iterativelySetFrameState(frame=self.trainTestEntryFrame, state=state)

    def iterativelySetFrameState(self, frame, state):
        if type(frame) == tk.LabelFrame or type(frame) == tk.Frame:
            for w in frame.winfo_children():
                self.iterativelySetFrameState(w, state)
        else:
            frame['state'] = state


    def updateTestVal(self, *args):
        try:
            val = int(self.trainIntVar.get())
        except ValueError:
            # messagebox.showinfo("Warning", "Train/test values should be integers")
            self.testIntVar.set('')
            return
        if val > 100:
            self.testIntVar.set('')
            # messagebox.showinfo("Warning", "Train/test values should be no less than 0 and no greater than 100")
            return
        self.testIntVar.set(100-val)

    def updateTrainVal(self, *args):
        try:
            val = int(self.testIntVar.get())
        except ValueError:
            # messagebox.showinfo("Warning", "Train/test values should be integers")
            self.trainIntVar.set('')
            return
        if val > 100:
            # messagebox.showinfo("Warning", "Train/test values should be no less than 0 and no greater than 100")
            self.trainIntVar.set('')
            return
        self.trainIntVar.set(100-val)
    
    def updateBestLoss(self, *args):
        modelDirName = self.modelDirSelVar.get()
        if modelDirName == '':
            self.bestLossLabel['text'] = 'Select a model directory'
            return
        
        configParser = ConfigParser()
        modelDir = os.path.join(self.lineDataDir, 'model_data', modelDirName)
        configParser.read(os.path.join(modelDir, '.model_info.ini'))
        self.modelDirLossFuncLabel['text'] = 'Loss function: ' + configParser['Model Directory Parameters']['fittinglossfunction']

        modelsDir = os.path.join(modelDir, 'matrix_params')
        if len(os.listdir(modelsDir)) == 0:
            self.numModelsLabel['text'] = 'Number models in this model directory: 0'
            self.bestLossLabel['text'] = 'No best loss'
            return
        configParser.read(os.path.join(modelsDir, 'best_result.txt'))
        bestLoss = configParser['ResultInfo']['trainlossvalue']
        self.numModelsLabel['text'] = 'Number models in this model directory: ' + str(len(os.listdir(modelsDir)))
        self.bestLossLabel['text'] = 'Best loss for this directory ('+configParser['ResultInfo']['numiterations']+' iterations): '+bestLoss
    #######################################################################################################################


    ################################################ Model Dir Creation #####################################################
    def createModelDir(self):

        # if len(entries) == 0:
        entries = self.collectModelDirEntries()
        if entries == None:
            return

        # Creates folder for storing model data
        if os.path.exists(os.path.join(self.lineDataDir, 'model_data')) == False:
            os.mkdir(os.path.join(self.lineDataDir, 'model_data'))
        # If a model dir with the user-given name exists, ask permission to overwrite it
        if os.path.exists(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'])):
            if messagebox.askokcancel('Overwrite existing model directory', 'You already have a model directory with the name '+entries['ModelName']+'. If you continue, this will be overwritten'):
                pass
            else:
                return
            
        if entries['CopyTrainJourneys'] == 'None' or entries['CopyTrainJourneys'] == '':
            specificJourneys = None
        else:
            reader = ConfigParser()
            reader.read(os.path.join(self.lineDataDir, 'model_data', entries['CopyTrainJourneys'], '.model_info.ini'))
            specificJourneys = reader['Model Directory Parameters']['trainjourneys'].split(',')
            entries = self.getModelDirParams(entries)
        
        probVectorSets, trainJourneys = self.pvc.createProbVectors(
                                                    arrivalsDir=os.path.join(self.lineDataDir, 'arrivals', entries['Route']),
                                                    departureTimeRange=entries['DepTimeRange'],
                                                    addIntersectingJourneys=entries['UnknownDepTimes'],
                                                    timeStep=entries['TimeStep'],
                                                    stopsOnly=entries['StopsOnly'],
                                                    trainSplit=entries['TrainSplit'],
                                                    specificJourneys=specificJourneys
                                                    )

        # Create new model directory only once probVectorSets has been computed, as computing it can take a while
        if os.path.exists(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'])):
            shutil.rmtree(os.path.join(self.lineDataDir, 'model_data', entries['ModelName']))
        os.mkdir(os.path.join(self.lineDataDir, 'model_data', entries['ModelName']))
        os.mkdir(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'], 'train_prob_vectors'))
        os.mkdir(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'], 'matrix_params'))

        # Stores the user-given parameters used in prob. vector computation in the '.model_info.ini' file for easy access later
        model_params = ConfigParser()
        model_params.add_section('Model Directory Parameters')
        for k in entries:
            if k == 'DepTimeRange':
                model_params['Model Directory Parameters']['DepTimeMin'] = entries[k][0].__str__()
                model_params['Model Directory Parameters']['DepTimeMax'] = entries[k][1].__str__()
            elif k == 'TimeStep':
                model_params['Model Directory Parameters']['TimeStep'] = str(entries[k].total_seconds())
            elif k == 'CopyTrainJourneys':
                continue
            else:
                model_params['Model Directory Parameters'][k] = entries[k].__str__()
        model_params['Model Directory Parameters']['NumJourneys'] = str(trainJourneys.shape[0])
        model_params['Model Directory Parameters']['TrainJourneys'] = ','.join(trainJourneys.index)
        model_params.write(open(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'], '.model_info.ini'), 'w'))

        # Store the train prob. vectors
        for k in probVectorSets:
            probVectorSets[k].to_parquet(os.path.join(self.lineDataDir, 'model_data', entries['ModelName'], 'train_prob_vectors', '-'.join(str(l) for l in k)))

        self.modelDirSelEntry['values'] = os.listdir(os.path.join(self.lineDataDir, 'model_data'))

    def collectModelDirEntries(self):
        errors = []
        
        # Use same train journeys as another set --------------
        copyTrainJourneys = self.copyTrainJourneysVar.get()
        # if copyTrainJourneys != 'None' and copyTrainJourneys != '':
        #     entries = self.getModelDirParams(copyTrainJourneys)
        #     return entries
        #--------------------------------------------------

        # Route ------------------------
        route = self.routeSelVar.get()
        #--------------------------------

        # Departure time range ---------------------------------------------------------------------------------
        try:
            depTimeLow = datetime.time(hour=int(self.depTimeLowHourVar.get()), minute=int(self.depTimeLowMinVar.get()))
            depTimeHigh = datetime.time(hour=int(self.depTimeHighHourVar.get()), minute=int(self.depTimeHighMinEntry.get()))
        #     if datetime.datetime.combine(datetime.date.min, depTimeLow) > datetime.datetime.combine(datetime.date.min, depTimeHigh):
        #         errors.append('Departure time minimum should be earlier than departure time maximum')
        except ValueError:
            errors.append('Departure time values must be integers')
        #-------------------------------------------------------------------------------------------------------

        # Include unknown dep. time journeys -------------
        includeUnkDepJourneys = self.unkDepTimeVar.get()
        #-------------------------------------------------

        # Train/test split ----------------------------------------
        try:
            trainValue = int(self.trainIntVar.get())
            testValue = int(self.testIntVar.get())
        except ValueError:
            errors.append('Train/test values should be integers')
        #----------------------------------------------------------

        # Time step -------------------------------------------------------------
        try:
            timeStep = datetime.timedelta(seconds=int(self.timeStepEntry.get()))
        except ValueError:
            errors.append('Time step seconds value should be an integer')
        #-----------------------------------------------------------------------

        # Stops only ------------------------
        stopsOnly = self.stopsOnlyBool.get()
        #------------------------------------

        # Model directory name ---------------------
        modelDirName = self.modelNameEntry.get()
        #-------------------------------------------

        # Loss Function -------------------
        lossFunction = self.lossFuncEntry.get()
        if lossFunction == '':
            messagebox.showinfo("Error", "Select a loss function")
            return
        #----------------------------------

        if len(errors) != 0:
            messagebox.showinfo('Error', ''.join('\n'+e for e in errors))
            return

        return {
            'CopyTrainJourneys': copyTrainJourneys,
            'Route': route,
            'DepTimeRange': (depTimeLow, depTimeHigh),
            'UnknownDepTimes': includeUnkDepJourneys,
            'TrainSplit': trainValue,
            'TimeStep': timeStep,
            'StopsOnly': stopsOnly,
            'ModelName': modelDirName,
            'FittingLossFunction': lossFunction
        }
    
    def getModelDirParams(self, entries):
        reader = ConfigParser()
        reader.read(os.path.join(self.lineDataDir, 'model_data', entries['CopyTrainJourneys'], '.model_info.ini'))
        entries['Route'] = reader['Model Directory Parameters']['route']
        minDepTime = dateutil.parser.parse(reader['Model Directory Parameters']['deptimemin']).time()
        maxDepTime = dateutil.parser.parse(reader['Model Directory Parameters']['deptimemax']).time()
        entries['DepTimeRange'] = [minDepTime, maxDepTime]
        entries['UnkownDepTimes'] = (reader['Model Directory Parameters']['unknowndeptimes'] == 'True')
        entries['TrainSplit'] = int(reader['Model Directory Parameters']['trainsplit'])

        return entries
    ##########################################################################################################################


    ############################################### Plot Train Vectors #######################################################
    def plotTrainVectors(self):
        self.continuePlotting = True
        try:
            minNumJourneys = int(self.minNumJourneysEntry.get())
        except ValueError:
            messagebox.showinfo("Error", "Min. Num Journeys should be an integer")
            return

        modelDir = self.modelDirSelEntry.get()
        if modelDir == '':
            messagebox.showinfo("Error", "Select a model directory")
            return

        plt.close()
        fig = plt.gcf()
        for f in os.listdir(os.path.join(self.lineDataDir, 'model_data', modelDir, 'train_prob_vectors')):
            keys = f.split('-')
            if int(keys[2]) < minNumJourneys:
                continue

            probVectorSet = pd.read_parquet(os.path.join(self.lineDataDir, 'model_data', modelDir, 'train_prob_vectors', f))
            for v in probVectorSet.index:
                if self.continuePlotting == False:
                    plt.close()
                    return
                fig.clear()
                ax = fig.gca()
                ax.plot(probVectorSet.loc[v])
                ax.set_title(v)
                plt.xticks(rotation=90)
                plt.ylim(-0.1, 1.1)
                plt.show(block=False)
                plt.pause(0.01)
        plt.close()
        self.continuePlotting = False

    def stopPlotting(self):
        self.continuePlotting = False
    ###########################################################################################################################


    ######################################### Plot normalized train journeys #############################################################
    def plotTrainJourneys(self):

        modelDir = self.modelDirSelEntry.get()
        if modelDir == '':
            messagebox.showinfo("Error", "Select a model directory")
            return

        configParser = ConfigParser()
        configParser.read(os.path.join(self.lineDataDir, 'model_data', modelDir, '.model_info.ini'))
        route = configParser['Model Directory Parameters']['route']
        trainJourneyIndicies = configParser['Model Directory Parameters']['trainjourneys'].split(',')
        arrivalTimesDF = pd.read_parquet(os.path.join(self.lineDataDir, 'arrivals', route))
        if 'JourneyCode' in arrivalTimesDF:
            arrivalTimesDF = arrivalTimesDF.drop('JourneyCode', axis=1)
        if 'VehicleRef' in arrivalTimesDF:
            arrivalTimesDF = arrivalTimesDF.drop('VehicleRef', axis=1)
        trainJourneys = arrivalTimesDF.loc[trainJourneyIndicies]

        trainJourneys = self.pvc.normalizeJourneys(journeys=trainJourneys)

        plt.close()
        fig = plt.gcf()
        ax = fig.gca()
        for c in trainJourneys.columns:
            for l in ax.lines:
                l.set_alpha(0.05)
            departingJourneys = trainJourneys[pd.isna(trainJourneys[c])==False]
            if departingJourneys.shape[0] == 0:
                continue
            for j in departingJourneys.index:
                ax.plot(trainJourneys.columns, [0 for n in range(trainJourneys.columns.get_loc(c))]+[t.total_seconds()/60 for t in trainJourneys.loc[j, c:]])
            ax.set_title("Departing from "+str(c))
            plt.ylabel("Arrival time (Minutes since departure)")
            plt.xlabel("Location")
            plt.xticks(rotation=90)
            plt.show(block=False)
            plt.pause(1)
            trainJourneys = trainJourneys.drop(departingJourneys.index, axis=0)
        plt.close()
    ##########################################################################################################################


    ############################################# Model fitting functions #####################################################
    def fitModel(self):

        modelDirectory = self.modelDirSelEntry.get()
        if modelDirectory == '':
            messagebox.showinfo("Error", "Please select a model directory to begin fitting")
            return
        try:
            maxNumIterations = int(self.maxNumIterationsEntry.get())
        except ValueError:
            messagebox.showinfo("Error", "Max. num. iterations Nelder-Mead must be an integer")
            return
        storeAllResults = self.storeAllResultsBool.get()
        
        if os.path.exists(os.path.join(self.lineDataDir, 'model_data', modelDirectory, 'matrix_params')) == False:
            os.mkdir(os.path.join(self.lineDataDir, 'model_data', modelDirectory, 'matrix_params'))

        t1 = Thread(target=lambda:self.mf.fitMatrix(modelDataDir=os.path.join(self.lineDataDir, 'model_data', modelDirectory), numIterationsPerNelderMead=maxNumIterations, storeAllResults=storeAllResults), daemon=True)
        t1.start()

        t2 = Thread(target=self.reportFittingProgress, daemon=True)
        t2.start()


    def reportFittingProgress(self):
        self.continueReporting = True
        while self.continueReporting == True:
            try:
                self.fittingProgressLabel['text'] = "Current loss (7dp): "+ str(self.mf.result['fun'].__round__(7)) +"\nNum Nelder-Mead iterations: " + str(self.mf.result['nit'])
                if self.mf.loopComplete == True:
                    self.updateBestLoss()
                    self.mf.loopComplete = False
            except AttributeError:
                pass
            time.sleep(1)
        # self.fittingProgressLabel['text'] = "Final loss: " + str(self.mf.result['fun'].__round__(5)) +"\nNum iterations: " + str(self.mf.result['nit'])
    
    def stopFitting(self):
        try:
            self.continueReporting = False
            self.fittingProgressLabel['text'] = "Final loss: " + str(self.mf.result['fun'].__round__(5)) +"\nNum iterations: " + str(self.mf.result['nit'])
            self.mf.continueFitting = False
            # t = Thread(target=time.sleep(1), daemon=True)
            # t.start()
            # self.updateBestLoss()
        except AttributeError:
            pass
        t = Thread(target=time.sleep(1), daemon=True)
        t.start()
        self.updateBestLoss()
    ###########################################################################################################################


    def collectModelFitEntries(self):
        errors = []

        modelDirectory = self.modelDirSelEntry.get()

        storeAllResults = self.storeAllResultsBool.get()

        return {'ModelDir': modelDirectory,
                'StoreAll': storeAllResults}
    
    # Plots journeys specified by user and computed model on the same axes
    def plotModel(self):
        entries = self.collectTrainPlotEntries()

        self.mf.plotActualProbVectorsAndModelProbVectors(matrixParams=entries[''])

    #---------------------------------------------------------------------------------------------------------------

    #----------------------------- Model evaluation ------------------------------------------------------------------
    # Gets journeys from arrivals DF as specified by user and computes loss between computed model and the prob. vectors computed from these journeys
    def beginEvaluation(self, *args):
        
        modelDir = self.modelDirSelEntry.get()
        if modelDir == '':
            messagebox.showinfo("Error", "Select a model directory")
            return
        # if os.path.exists(os.path.join(self.lineDataDir, 'model_data', modelDir)) == False:
        #     messagebox.showinfo("Error", "Select a valid model directory")
        #     return

        if len(os.listdir(os.path.join(self.lineDataDir, 'model_data', modelDir, 'matrix_params'))) == 0:
            messagebox.showinfo("Error", "This model directory has no matrix parameters - use the model fitting button to fit some")
            return
        
        self.evaluationNode = MenuNode(parent=self.menu)
        self.evaluationMenu = EvaluatorMenu(menu=self.evaluationNode, lineDataDir=self.lineDataDir, modelDirName=modelDir)
        self.evaluationMenu.initialize()
    #-----------------------------------------------------------------------------------------------------------------




# class ModelViewer():

#     def __init__(self, menu, probabilityVectors, testProbabilityVectors, randomWalkModeller, arrivalTimes):

#         self.modelViewerWindow = menu.window
#         self.randomWalkModeller = randomWalkModeller
#         self.probabilityVectors = probabilityVectors
#         self.testProbabilityVectors = testProbabilityVectors
#         self.trainArrivals = arrivalTimes[0]
#         self.testArrivals = arrivalTimes[1]
#         self.arrivalTimes = arrivalTimes[2]
        

#         # self.modelViewerWindow = tk.Tk()
#         self.modelViewerWindow.title("Model Viewer")

#         self.exitButton = ttk.Button(text="Exit", master=self.modelViewerWindow, command=self.killWindow)
#         self.exitButton.pack(side="bottom", anchor="se")

#         self.welcomeLabel = ttk.Label(text="This is the model viewer\nYou can plot the probability vectors, or fit the model to the vectors", master=self.modelViewerWindow)
#         self.welcomeLabel.pack()

#         # self.thresholdFrame = ttk.Frame(master=self.modelViewerWindow)
#         # self.thresholdFrame.pack(side="top")
#         # self.thresholdLabel = ttk.Label(text="Choose a probability threshold", master=self.thresholdFrame)
#         # self.thresholdLabel.pack(side="left")
#         # self.thresholdEntry = ttk.Spinbox(from_=0, to=1, increment=0.01, master=self.thresholdFrame)
#         # self.thresholdEntry.pack(side="left")

#         # self.storeFrame = ttk.Frame(master=self.modelViewerWindow)
#         # self.storeFrame.pack(side="top")
#         # self.storeDirLabel = ttk.Label(text="Storage directory", master=self.storeFrame)
#         # self.storeDirLabel.pack(side="left")
#         # self.storeDir = ttk.Entry(master=self.storeFrame)
#         # self.storeDir.pack(side="right")

#         # self.storeProbVectorsButton = ttk.Button(text="Store probability vectors", master=self.modelViewerWindow, command=self.storeData)
#         # self.storeProbVectorsButton.pack()

#         self.fitButtonsFrame = ttk.Frame(master=self.modelViewerWindow)
#         self.fitButtonsFrame.pack(side="top")
#         self.fitTransitionMatrixButton = ttk.Button(text="Begin fitting transition matrix", master=self.fitButtonsFrame, command=self.startFitting)
#         self.fitTransitionMatrixButton.pack(side="left")
#         self.stopFittingMatrixButton = ttk.Button(text="Stop fitting the matrix", master=self.fitButtonsFrame, command=self.stopFitting)
#         self.stopFittingMatrixButton.pack(side="left")

#         self.fitterProgressFrame = ttk.Frame(master=self.modelViewerWindow)
#         self.fitterProgressFrame.pack(side="top")
#         self.progress = ''
#         self.getProgress = False
#         self.progressLabel = ttk.Label(text="Start running the model", master=self.fitterProgressFrame)
#         self.progressLabel.pack(side="top")
#         self.bestResultLabel = ttk.Label(text="The best result will be displayed here", master=self.fitterProgressFrame)
#         self.bestResultLabel.pack(side="top")
#         self.curResultLabel = ttk.Label(text="The result for the most recent run will be displayed here", master=self.fitterProgressFrame)
#         self.curResultLabel.pack(side="top")

#         # self.viewModelButton = ttk.Button(text="View best model on plot with probability vectors", command=lambda:self.plotModel(self.bestResult, self.probabilityVectors), master=self.fitterFrame)

#         #---------------------------------------- Model saving -------------------------------------------------------------------
#         self.modelSavingFrame = ttk.Frame(master=self.modelViewerWindow)
#         self.modelSavingFrame.pack(side="top")
#         self.saveModelButton = ttk.Button(text="Save model", command=self.storeModel, master=self.modelSavingFrame)
#         self.modelDir = ttk.Entry(master=self.modelSavingFrame)
#         self.modelDir.insert(index=0, string='./BusModeller/data/random_walk_modeller_cache/currentModel')
#         #-------------------------------------------------------------------------------------------------------------------------

#         #--------------------------------------- Model plotting ------------------------------------------------------------------
#         self.plotModelFrame = ttk.Frame(master=self.modelViewerWindow)
#         # self.plotModelFrame.pack(side="bottom")
#         self.viewProbVectorsButton = ttk.Button(text="Plot probability vectors and model", master=self.plotModelFrame, command=self.plot)
#         self.viewProbVectorsButton.pack(side="top")

#         self.longTermPlotBool = IntVar()
#         self.plotShortTermPrediction = ttk.Checkbutton(text="Plot long term prediction", variable=self.longTermPlotBool, command=lambda:self.toggleVar(self.longTermPlotBool), master=self.plotModelFrame)
#         self.plotShortTermPrediction.pack(side="top")
#         self.shortTermPlotBool = IntVar()
#         self.plotShortTermPrediction = ttk.Checkbutton(text="Plot short term prediction", variable=self.shortTermPlotBool, command=lambda:self.toggleVar(self.shortTermPlotBool), master=self.plotModelFrame)
#         self.plotShortTermPrediction.pack(side="top")

#         self.closePlotButton = ttk.Button(text="Stop plot", master=self.plotModelFrame, command=self.closePlot)
#         self.closePlotButton.pack(side="top")

#         # self.plotProgressLabel = ttk.Label(master=self.plotModelFrame)
#         # self.plotProgressLabel.pack(side="bottom")
#         #------------------------------------------------------------------------------------------------------------------------------

#         self.evaluateModelButton = ttk.Button(text="Evaluate model", command=self.evaluateModel, master=self.modelViewerWindow)

#         self.modelViewerWindow.mainloop()

#     def killWindow(self):
#         self.modelViewerWindow.destroy()

#     def startFitting(self):
#         self.progressLabel['text'] = 'Nelder-Mead will started running soon...'
#         self.run = True
#         self.progressReporting = True
#         t = Thread(target=self.fitMatrix, daemon=True)
#         t.start()

#         t1 = Thread(target=self.progressLoop, daemon=True)
#         t1.start()

#     def stopFitting(self):
#         self.randomWalkModeller.stopComputing()
#         self.run = False
#         self.progressReporting = False

#     def fitMatrix(self):

#         while self.run:
#             self.running = True

#             self.result = self.randomWalkModeller.fitTransitionMatrix(probabilityVectors=self.probabilityVectors
#                                                                     #   store=False
#                                                                     )
            
#             try:
#                 if self.result[1] < self.bestResult[1]:
#                     self.bestResult = self.result
#                     self.bestResultLabel['text'] = 'Best result loss: ' + str(self.bestResult[1])
#                     self.curResultLabel['text'] = "Previous iteration loss: " + str(self.result[1])
#                 else:
#                     self.curResultLabel['text'] = "Previous iteration loss: " + str(self.result[1])
#             except AttributeError:
#                 self.bestResult = self.result
#                 print(self.bestResult[1])
#                 self.bestResultLabel['text'] = 'Best result loss: ' + str(self.bestResult[1])
#                 self.curResultLabel['text'] = "Previous iteration loss: " + str(self.result[1])
            

#         self.running = False
#         if self.result == False:
#             self.progressLabel['text'] = 'Nelder-Mead forced to end'
#         else:
#             self.progressLabel['text'] = 'Nelder-Mead ended'
#         self.plotModelFrame.pack(side="bottom")

#         self.saveModelButton.pack(side="left")
#         self.modelDir.pack(side="left")

#         self.evaluateModelButton.pack(side="bottom")

#     def progressLoop(self):
#         while self.progressReporting:
#             self.modelViewerWindow.after(5000, self.reportProgress)
#         self.progressLabel['text'] = 'Nelder-Mead will end soon...'

#     def reportProgress(self):
#         if self.progressReporting == False:
#             return
#         loopNum, currentLoss = self.randomWalkModeller.getProgress()
#         try:
#             if loopNum == prevLoopNum:
#                 return
#         except NameError:
#             pass

#         try:
#             if loopNum % 500 == 0:
#                 self.progressLabel['text'] = 'Loop ' + str(loopNum) + " Loss is " + str(currentLoss)
#         except AttributeError:
#             self.progressLabel['text'] = 'Nelder-Mead should begin soon...'

#         prevLoopNum = loopNum
#         return

#         # self.modelViewerWindow.after(1000, self.reportProgress)
#         # self.progressLabel['text'] = 'Model will end after the next loop is done'

#     def toggleVar(self, var):
#         var.set(not var.get())

#     def plot(self):
        
#         if self.running == True and self.run == False:
#             self.plotProgressLabel['text'] = 'Wait until Nelder-Mead stops running to plot'
#             return
#         elif self.running == True and self.run == True:
#             self.plotProgressLabel['text'] = 'Stop Nelder-Mead to plot the model'
#             return
        
#         self.plotBool = True
#         while self.plotBool == True:
#             self.randomWalkModeller.startPlotting(self.bestResult[0], self.probabilityVectors, self.longTermPlotBool.get(), self.shortTermPlotBool.get())

#     def closePlot(self):
#         self.plotBool = False
#         self.randomWalkModeller.stopPlotting()

#     def storeModel(self):
#         dir = self.modelDir.get()
#         if dir == '':
#             self.statusLabel = ttk.Label(text="Enter a valid directory", master=self.modelSavingFrame)
#             self.statusLabel.pack(side="left")
#             return
#         with open(dir, 'wb') as f:
#             pickle.dump((self.bestResult, self.probabilityVectors, self.testProbabilityVectors, (self.trainArrivals, self.testArrivals, self.arrivalTimes)), f)
#         self.statusLabel = ttk.Label(text="Model stored at specified directory", master=self.modelSavingFrame)
#         self.statusLabel.pack(side="left")
        
#     def evaluateModel(self):
#         evaluator = ModelEvaluator(params=self.bestResult[0], testProbVectors=self.testProbabilityVectors, testArrivalTimes=self.testArrivals, departureTime=self.probabilityVectors[0][0])
#         # evaluator = ModelEvaluator(self.bestResult[0], self.testProbabilityVectors)


# class ModelEvaluator():

#     def __init__(self, params, testProbVectors, testArrivalTimes, departureTime):
#         self.evaluator = RandomWalkModeller.ModelEvaluator(modelParams=params, testProbVectors=testProbVectors, testArrivalTimes=testArrivalTimes)
#         self.probVectors = testProbVectors
#         self.params = params

#         self.evaluatorWindow = tk.Tk()

#         self.specificProbFrame = ttk.Frame(master=self.evaluatorWindow)
#         self.specificProbFrame.pack(side="top")
#         self.computeSpecificProbButton = ttk.Button(text="Compute probability of bus arriving at at stop at time, given departure time"+str(departureTime), command=self.computeSpecificProbArrival, master=self.specificProbFrame)
#         self.computeSpecificProbButton.pack(side="left")
#         self.stopLabel = ttk.Label(text="Stop", master=self.specificProbFrame)
#         self.stopLabel.pack(side="left")
#         self.stopCombo = ttk.Combobox(values=testArrivalTimes.columns, master=self.specificProbFrame)
#         self.stopCombo.pack(side="left")

#         self.allArrivalsFrame = ttk.Frame(master=self.evaluatorWindow)
#         self.allArrivalsFrame.pack(side="top")
#         self.computeArrivalsAllStops = ttk.Button(text="Compute loss wrt to test set", command=self.computeLoss, master=self.allArrivalsFrame)
#         self.computeArrivalsAllStops.pack(side="left")
#         self.arrivalTimeProbabilityThreshold = ttk.Spinbox(from_=0, to=1, increment=0.01, master=self.allArrivalsFrame)
#         self.arrivalTimeProbabilityThreshold.pack(side="left")

#         self.evaluatorWindow.mainloop()


#     def computeLoss(self):
#         self.evaluator.compareModelToTestSet()

#     def computeSpecificProbArrival(self):
#         pass

#     def computeAllArrivals(self):
#         pass


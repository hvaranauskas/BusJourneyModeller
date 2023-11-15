from .MockDataGenerator import MockDataGenerator

import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import os
import pandas as pd

from threading import Thread

from configparser import ConfigParser, ExtendedInterpolation


# Module is a front-end which takes user input from a user interface and feeds it into back-end MockDataGenerator to produce mock data
class MockDataGeneratorMenu():

    def __init__(self, menu, lineDataDir):
        self.mdg = MockDataGenerator()
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('configs/default.ini')
        self.menu = menu
        self.mainWindow = self.menu.window
        self.lineDataDir = lineDataDir
        self.widgetStates = ['disabled', 'normal']

    def initialize(self):

        #---------------------------- Initialization ----------------------------------------
        self.mainWindow.title("MockDataGenerator")
        #-----------------------------------------------------------------------------------

        gapx = 5
        borderx = 10
        gapy = 5

        #------------------------------------ Welcome ---------------------------------------------------------------------------------------
        self.welcomeFrame = tk.Frame(master=self.mainWindow)
        self.welcomeFrame.pack(side="top")
        self.welcomeLabel = tk.Label(text="Mock Data Generator", master=self.welcomeFrame)
        self.welcomeLabel.pack(padx=gapx, pady=gapy)
        #------------------------------------------------------------------------------------------------------------------------------------

        #------------------------------------ Route parameters --------------------------------------------
        self.routeParamsFrame = tk.LabelFrame(text="Route Parameters", master=self.mainWindow)
        self.routeParamsFrame.pack(side="top", fill='x', padx=borderx, pady=gapy)

        self.journeyDistancesLabel = tk.Label(text="Distances between stops (meters)", master=self.routeParamsFrame)
        self.journeyDistancesLabel.pack(side="top", anchor="w")
        self.journeyDistancesEntry = tk.Entry(master=self.routeParamsFrame)
        self.journeyDistancesEntry.pack(side="top", fill='x', padx=gapx, pady=(0, gapy))
        self.journeyDistancesEntry.insert(index=0, string=self.config['MockDataGenerator']['journeydistances'])

        self.departTimesLabel = tk.Label(text="Departure Times (minutes)", master=self.routeParamsFrame)
        self.departTimesLabel.pack(side="top", anchor="w")
        self.departureTimes = tk.Entry(master=self.routeParamsFrame)
        self.departureTimes.pack(side="top", fill='x', padx=gapx, pady=(0, gapy))
        self.departureTimes.insert(index=0, string=self.config['MockDataGenerator']['departuretimes'])
        #-----------------------------------------------------------------------------------------------------

        #----------------------------------- World Parameters -------------------------------------------------------
        self.worldParamsFrame = tk.LabelFrame(text="World Parameters", master=self.mainWindow)
        self.worldParamsFrame.pack(side="top", fill='x', padx=borderx, pady=gapy)

        self.daysInWeekLabel = tk.Label(text="Num. days in a week:", master=self.worldParamsFrame)
        self.daysInWeekLabel.grid(column=0, row=0, sticky='E', padx=gapx, pady=gapy/4)
        self.daysInWeekEntry = tk.Spinbox(from_=1, to_=1000, width=5, increment=1, master=self.worldParamsFrame)
        self.daysInWeekEntry.grid(column=1, row=0, padx=gapx, pady=gapy/4)
        self.daysInWeekEntry.delete(0)
        self.daysInWeekEntry.insert(index=0, s=self.config['MockDataGenerator']['numdaysinweek'])

        self.trafficSimilarityLabel = tk.Label(text="Traffic similarity interval (minutes):", master=self.worldParamsFrame)
        self.trafficSimilarityLabel.grid(column=0, row=1, sticky='E', padx=gapx, pady=gapy/4)
        self.trafficSimilarityEntry = tk.Spinbox(from_=0, to_=9999, width=5, increment=1, master=self.worldParamsFrame)
        self.trafficSimilarityEntry.grid(column=1, row=1, padx=gapx, pady=gapy/4)
        self.trafficSimilarityEntry.delete(0)
        self.trafficSimilarityEntry.insert(index=0, s=self.config['MockDataGenerator']['trafficsimilarityinterval'])

        self.maxSpeedLabel = tk.Label(text="Bus max travel speed (meters/second):", master=self.worldParamsFrame)
        self.maxSpeedLabel.grid(column=0, row=2, sticky='E', padx=gapx, pady=gapy/4)
        self.maxSpeedEntry = tk.Spinbox(from_=0, to_=9999, width=5, increment=0.1, master=self.worldParamsFrame)
        self.maxSpeedEntry.grid(column=1, row=2, padx=gapx, pady=gapy/4)
        self.maxSpeedEntry.delete(0)
        self.maxSpeedEntry.insert(index=0, s=self.config['MockDataGenerator']['busmaxtravelspeed'])

        self.boardConstantLabel = tk.Label(text="Per-person boarding time (seconds):", master=self.worldParamsFrame)
        self.boardConstantLabel.grid(column=0, row=3, sticky='e', padx=gapx, pady=gapy/4)
        self.boardConstantEntry = tk.Spinbox(from_=1, to_=9999, increment=1, width=5, master=self.worldParamsFrame)
        self.boardConstantEntry.grid(column=1, row=3, padx=gapx, pady=gapy/4)
        self.boardConstantEntry.delete(0)
        self.boardConstantEntry.insert(index=0, s=self.config['MockDataGenerator']['perpersonboardingtime'])

        self.stopTimeLabel = tk.Label(text="Bus approach/recovery time (seconds):", master=self.worldParamsFrame)
        self.stopTimeLabel.grid(column=0, row=4, sticky='e', padx=gapx, pady=gapy/4)
        self.stopTimeEntry = tk.Spinbox(from_=1, to_=9999, width=5, master=self.worldParamsFrame)
        self.stopTimeEntry.grid(column=1, row=4, padx=gapx, pady=gapy/4)
        self.stopTimeEntry.delete(0)
        self.stopTimeEntry.insert(index=0, s=self.config['MockDataGenerator']['approachrecoverytime'])
        #-----------------------------------------------------------------------------------------------------

        #-------------------------------------- Traffic Simulation parameters -------------------------------------
        self.delayParamsFrame = tk.LabelFrame(text="Delay Randomness Parameters", master=self.mainWindow)
        self.delayParamsFrame.pack(side="top", fill='x', padx=borderx, pady=gapy)
        # self.delayParamsFrame.columnconfigure(3)

        self.simDelaysLabel = tk.Label(text="Simulate delays", master=self.delayParamsFrame)
        self.simDelaysLabel.grid(column=0, row=0, sticky='E')
        self.simDelaysVar = tk.BooleanVar()
        self.simDelaysVar.set(self.config['MockDataGenerator']['simulatedelays'])
        self.simDelaysVar.trace(mode='w', callback=self.toggleDelayParams)
        self.simDelaysEntry = tk.Checkbutton(variable=self.simDelaysVar, master=self.delayParamsFrame)
        self.simDelaysEntry.grid(column=1, row=0, sticky='W')

        self.constantDelaysLabel = tk.Label(text="Constant delays", master=self.delayParamsFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.constantDelaysLabel.grid(column=0, row=1, sticky='E')
        self.constantDelaysVar = tk.BooleanVar()
        self.constantDelaysVar.set(self.config['MockDataGenerator']['constantdelays'])
        self.constantDelaysEntry = tk.Checkbutton(variable=self.constantDelaysVar, master=self.delayParamsFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.constantDelaysEntry.grid(column=1, row=1, sticky='W')

        self.burrCParamLabel = tk.Label(text="Burr C Param range:", master=self.delayParamsFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrCParamLabel.grid(column=0, row=2, sticky='E')
        self.burrCLowParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.burrCLowParamFrame.grid(column=1, row=2)
        self.burrCLowParamLabel = tk.Label(text="Low:", master=self.burrCLowParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrCLowParamLabel.pack(side="left")
        self.burrCLowParamEntry = tk.Spinbox(from_=0, to_=9999, increment=0.1, width=4, master=self.burrCLowParamFrame)
        self.burrCLowParamEntry.delete(0, "end")
        self.burrCLowParamEntry.insert(index=0, s=self.config['MockDataGenerator']['burrclowparam'])
        self.burrCLowParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.burrCLowParamEntry.pack(side="left")
        self.burrCHighParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.burrCHighParamFrame.grid(column=2, row=2)
        self.burrCHighParamLabel = tk.Label(text="High:", master=self.burrCHighParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrCHighParamLabel.pack(side="left")
        self.burrCHighParamEntry = tk.Spinbox(from_=0, to_=9999, increment=0.1, width=4, master=self.burrCHighParamFrame)
        self.burrCHighParamEntry.delete(0, "end")
        self.burrCHighParamEntry.insert(index=0, s=self.config['MockDataGenerator']['burrchighparam'])
        self.burrCHighParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.burrCHighParamEntry.pack(side="left")

        self.burrDParamLabel = tk.Label(text="Burr D Param range:", master=self.delayParamsFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrDParamLabel.grid(column=0, row=3, sticky='E')
        self.burrDLowParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.burrDLowParamFrame.grid(column=1, row=3)
        self.burrDLowParamLabel = tk.Label(text="Low:", master=self.burrDLowParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrDLowParamLabel.pack(side="left")
        self.burrDLowParamEntry = tk.Spinbox(from_=0, to_=9999, increment=1, width=4, master=self.burrDLowParamFrame)
        self.burrDLowParamEntry.delete(0)
        self.burrDLowParamEntry.insert(index=0, s=self.config['MockDataGenerator']['burrdlowparam'])
        self.burrDLowParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.burrDLowParamEntry.pack(side="left")
        self.burrDHighParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.burrDHighParamFrame.grid(column=2, row=3)
        self.burrDHighParamLabel = tk.Label(text="High:", master=self.burrDHighParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.burrDHighParamLabel.pack(side="left")
        self.burrDHighParamEntry = tk.Spinbox(from_=0, to_=99999, increment=1, width=4, master=self.burrDHighParamFrame)
        self.burrDHighParamEntry.delete(0)
        self.burrDHighParamEntry.insert(index=0, s=self.config['MockDataGenerator']['burrdhighparam'])
        self.burrDHighParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.burrDHighParamEntry.pack(side="left")

        self.poissonParamLabel = tk.Label(text="Poisson Lambda range:", master=self.delayParamsFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.poissonParamLabel.grid(column=0, row=4, sticky='E')
        self.poissonLowParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.poissonLowParamFrame.grid(column=1, row=4)
        self.poissonLowParamLabel = tk.Label(text="Low:", master=self.poissonLowParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.poissonLowParamLabel.pack(side="left")
        self.poissonLowParamEntry = tk.Spinbox(from_=0, to_=9999, increment=1, width=4, master=self.poissonLowParamFrame)
        self.poissonLowParamEntry.delete(0)
        self.poissonLowParamEntry.insert(index=0, s=self.config['MockDataGenerator']['poissonlowparam'])
        self.poissonLowParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.poissonLowParamEntry.pack(side="left")
        self.poissonHighParamFrame = tk.Frame(master=self.delayParamsFrame)
        self.poissonHighParamFrame.grid(column=2, row=4)
        self.poissonHighParamLabel = tk.Label(text="High:", master=self.poissonHighParamFrame, state=self.widgetStates[self.simDelaysVar.get()])
        self.poissonHighParamLabel.pack(side="left")
        self.poissonHighParamEntry = tk.Spinbox(from_=0, to_=9999, increment=1, width=4, master=self.poissonHighParamFrame)
        self.poissonHighParamEntry.delete(0)
        self.poissonHighParamEntry.insert(index=0, s=self.config['MockDataGenerator']['poissonhighparam'])
        self.poissonHighParamEntry['state'] = self.widgetStates[self.simDelaysVar.get()]
        self.poissonHighParamEntry.pack(side="left")
        #---------------------------------------------------------------------------------------------------------------------

        #--------------------------------------- Generation parameters -----------------------------------------------------------
        self.generationParamsFrame = tk.LabelFrame(text="Generation Parameters", master=self.mainWindow)
        self.generationParamsFrame.pack(side="top", fill='x', padx=borderx, pady=gapy)

        self.simulationDaysLabel = tk.Label(text="Number of days to simulate", master=self.generationParamsFrame)
        self.simulationDaysLabel.grid(column=0, row=0, sticky='E')
        self.simulationDaysEntry = tk.Spinbox(from_=1, to_=99999, width=5, increment=1, master=self.generationParamsFrame)
        self.simulationDaysEntry.grid(column=1, row=0, sticky='W')
        self.simulationDaysEntry.delete(0)
        self.simulationDaysEntry.insert(index=0, s=self.config['MockDataGenerator']['simulationdays'])

        self.arrivalsNameLabel = tk.Label(text="Arrival Times Set Name", master=self.generationParamsFrame)
        self.arrivalsNameLabel.grid(column=0, row=1, sticky='E')
        self.arrivalsNameEntry = tk.Entry(master=self.generationParamsFrame)
        self.arrivalsNameEntry.grid(column=1, row=1, sticky='W')
        self.arrivalsNameEntry.insert(index=0, string='MockArrivals')
        #------------------------------------------------------------------------------------------------------------

        #------------------------------------------ Generation section -------------------------------------------------------
        self.generationFrame = tk.Frame(master=self.mainWindow)
        self.generationFrame.pack(side="top")
        self.generateDataButton = tk.Button(text="Generate Data", command=self.startGenerating, master=self.generationFrame)
        self.generateDataButton.pack(side="left")
        self.stopGeneratingButton = tk.Button(text="Stop generating data", command=self.stopGenerating, master=self.generationFrame)
        self.resultLabel = tk.Label(master=self.generationFrame)
        self.resultLabel.pack(side="left")
        #--------------------------------------------------------------------------------------------------------------------

    
        #--------------------------------------------- Arrival time plotting --------------------------------------------------
        self.plotFrame = tk.LabelFrame(text="Arrival Time Plotting", master=self.mainWindow)
        self.plotFrame.pack(side="top", fill='x', padx=borderx, pady=gapy)

        self.arrivalSetSelLabel = tk.Label(text="Arrival Times Set", master=self.plotFrame)
        self.arrivalSetSelLabel.grid(column=0, row=0, sticky='E')
        self.arrivalSetSelVar = tk.StringVar()
        self.arrivalSetSelVar.trace(mode='w', callback=self.updateNumJourneys)
        self.arrivalSetSelEntry = ttk.Combobox(values=os.listdir(os.path.join(self.lineDataDir, 'arrivals')), state='readonly', textvariable=self.arrivalSetSelVar, master=self.plotFrame)
        self.arrivalSetSelEntry.grid(column=1, row=0, sticky='W')
        self.arrivalSetSelNumJourneys = tk.Label(text="Num. journeys in arrivals set displayed here", master=self.plotFrame)
        self.arrivalSetSelNumJourneys.grid(column=0, columnspan=2, row=1)

        self.depTimeRangeLabel = tk.Label(text="Dep. time range (minutes)", master=self.plotFrame)
        self.depTimeRangeLabel.grid(column=0, row=2, sticky='E')
        self.depTimeRangeLowFrame = tk.Frame(master=self.plotFrame)
        self.depTimeRangeLowFrame.grid(column=1, row=2, sticky='W')
        self.depTimeRangeLowLabel = tk.Label(text="Low", master=self.depTimeRangeLowFrame)
        self.depTimeRangeLowLabel.pack(side="left")
        self.depTimeRangeLowEntry = tk.Spinbox(from_=0, to_=99, width=4, master=self.depTimeRangeLowFrame)
        self.depTimeRangeLowEntry.pack(side="left")
        self.depTimeRangeLowEntry.delete(0, 'end')
        self.depTimeRangeLowEntry.insert(index=0, s=self.config['MockDataGenerator']['plotdeptimelow'])
        self.depTimeRangeHighFrame = tk.Frame(master=self.plotFrame)
        self.depTimeRangeHighFrame.grid(column=2, row=2, sticky='W')
        self.depTimeRangeHighLabel = tk.Label(text="High", master=self.depTimeRangeHighFrame)
        self.depTimeRangeHighLabel.pack(side="left")
        self.depTimeRangeHighEntry = tk.Spinbox(from_=0, to_=99, increment=1, width=4, master=self.depTimeRangeHighFrame)
        self.depTimeRangeHighEntry.pack(side="left")
        self.depTimeRangeHighEntry.delete(0, 'end')
        self.depTimeRangeHighEntry.insert(index=0, s=self.config['MockDataGenerator']['plotdeptimehigh'])

        self.plotButton = tk.Button(text="Plot arrival times", command=self.plotArrivalTimes, master=self.plotFrame)
        self.plotButton.grid(column=0, row=3, columnspan=3)
        #--------------------------------------------------------------------------------------------------------------------

        self.mainWindow.mainloop()


    def startGenerating(self):

        entries = self.collectEntries()
        if entries == None:
            return
        else:
            self.stopGeneratingButton.pack(side="left")
            t1 = Thread(target=lambda:self.generateMockData(entries), daemon=True)
            t1.start()

            self.continueReporting = True
            t2 = Thread(target=self.reportProgress, daemon=True)
            t2.start()


    def collectEntries(self):

        error = False
        errors = []

        # Journey distances -----------------------------------------------------------------------
        try:
            strJourneyDistances = self.journeyDistancesEntry.get().split(" ")
            journeyDistances = [int(s) for s in strJourneyDistances]
            if len(journeyDistances) == 0:
                error = True
                errors.append("You should have at least one journey distance")
        except ValueError:
            error = True
            errors.append("Journey distances should be a sequence of integers separated by spaces")
        #---------------------------------------------------------------------------------------

        # Departure times --------------------------------------------------------------------------
        try:
            strDepartureTimes = self.departureTimes.get().split(" ")
            departureTimes = sorted([int(s) for s in strDepartureTimes])
            if len(departureTimes) == 0:
                error = True
                errors.append("You should have at least one departure time")
        except ValueError:
            error = True
            errors.append("Departure times should be a sequence of integers separated by spaces")
        #---------------------------------------------------------------------------------------------

        # Days in week -------------------------------------------------
        try:
            daysInWeek = int(self.daysInWeekEntry.get())
        except ValueError:
            error = True
            errors.append("Days in week should be an integer")
        #--------------------------------------------------------------

        # Traffic similarity interval ---------------------------------------------
        try:
            trafficSimilarityInterval = datetime.timedelta(minutes = int(self.trafficSimilarityEntry.get()))
        except ValueError:
            error = True
            errors.append("Traffic similarity interval should be an integer")
        #-----------------------------------------------------------------------

        # Max cruise speed --------------------------------------------------------------------------
        try:
            baseVelocity = float(self.maxSpeedEntry.get())
        except ValueError:
            error = True
            errors.append("Max speed should be a float")
        #---------------------------------------------------------------------------------------------

        # Boarding time --------------------------------------------------------------------------
        try:
            boardingTime = float(self.boardConstantEntry.get())
        except ValueError:
            error = True
            errors.append("Boarding time should be a float")
        #---------------------------------------------------------------------------------------------

        # Bus stopping time --------------------------------------------------------------------------
        try:
            stopTime = float(self.stopTimeEntry.get())
        except ValueError:
            error = True
            errors.append("Stop time should be a float")
        #---------------------------------------------------------------------------------------------

        # Delay parameters ------------------------------------------------------------------------------
        if self.simDelaysVar.get() == True:

            constantDelays = self.constantDelaysVar.get()

            # Distribution parameters ---------------------------------------------------------------------
            try:
                burrCLowParam = float(self.burrCLowParamEntry.get())
                burrCHighParam = float(self.burrCHighParamEntry.get())
                burrDLowParam = float(self.burrDLowParamEntry.get())
                burrDHighParam = float(self.burrDHighParamEntry.get())
                poissonLowParam = float(self.poissonLowParamEntry.get())
                poissonHighParam = float(self.poissonHighParamEntry.get())
            except ValueError:
                error = True
                errors.append("Delay parameters should be floats")
            #---------------------------------------------------------------------------------------------

        else:
            constantDelays = False
            burrCLowParam = None
            burrCHighParam = None
            burrDLowParam = None
            burrDHighParam = None
            poissonLowParam = None
            poissonHighParam = None
        #------------------------------------------------------------------------------------------------

        # Simulation days -------------------------------------------------------------------------
        try:
            simulationDays = int(self.simulationDaysEntry.get())
        except ValueError:
            error = True
            errors.append("Number of simulation days should be an integer")
        #------------------------------------------------------------------------------------------

        # Arrivals name ----------------------------------------------------------------
        arrivalsName = self.arrivalsNameEntry.get()
        if os.path.exists(os.path.join(self.lineDataDir, 'arrivals', arrivalsName)):
            if messagebox.askokcancel("Arrivals already exists", "A set of mock arrival times with the name '"+arrivalsName+"' already exists, overwrite it?")==False:
                return
            else:
                os.remove(os.path.join(self.lineDataDir, 'arrivals', arrivalsName))
        #------------------------------------------------------------------------------


        # Report errors to user if there are any
        if error == True:
            errorMsg = 'Error:'
            for e in errors:
                errorMsg += '\n'+e
            messagebox.showinfo(title="Invalid input", message=errorMsg, master=self.mainWindow)
            return
        
        return {'JourneyDistances': journeyDistances, 
                'DepTimes': departureTimes, 
                'DaysInWeek': daysInWeek, 
                'TraffSimInterval': trafficSimilarityInterval, 
                'BaseVelocity': baseVelocity,
                'BoardTime': boardingTime, 
                'StopTime': stopTime, 
                'SimDelays': self.simDelaysVar.get(),
                # constantDelays,
                'Burr': [(burrCLowParam, burrCHighParam), (burrDLowParam, burrDHighParam)], 
                'Poisson': (poissonLowParam, poissonHighParam), 
                'NumDays': simulationDays,
                'ArrivalsName': arrivalsName
                }


    def generateMockData(self, entries):

        if os.path.exists(os.path.join(self.lineDataDir, 'arrivals')) == False:
            os.mkdir(os.path.join(self.lineDataDir, 'arrivals'))

        self.arrivalTimes = self.mdg.createMockData(
                                                    journeyDistances=entries['JourneyDistances'],
                                                    depTimes=entries['DepTimes'], 
                                                    daysInWeek=entries['DaysInWeek'],
                                                    trafficSimilarityInterval=entries['TraffSimInterval'],
                                                    baseVelocity=entries['BaseVelocity'],
                                                    boardingTimeConstant=entries['BoardTime'],
                                                    stopTimeConstant=entries['StopTime'],
                                                    simDelays=entries['SimDelays'],
                                                    burrParamRange=entries['Burr'],
                                                    poissonParamRange=entries['Poisson'],
                                                    simulationDays=entries['NumDays'],
                                                    # storageDir=os.path.join(self.lineDataDir, 'arrivals', 'MockRoute')
                                                    )
        
        if os.path.exists(os.path.join(self.lineDataDir, 'arrivals')) == False:
            os.mkdir(os.path.join(self.lineDataDir, 'arrivals'))

        self.arrivalTimes.to_parquet(os.path.join(self.lineDataDir, 'arrivals', entries['ArrivalsName']))

        self.continueReporting = False
        self.resultLabel['text'] = "Finished generating data"

        # self.plotButton.pack(side="top")
        # self.changeStateFrame(state='normal', frame=self.plotFrame)
        self.arrivalSetSelEntry['values'] = os.listdir(os.path.join(self.lineDataDir, 'arrivals'))


    def reportProgress(self):

        while self.continueReporting == True:
            self.resultLabel['text'] = 'Simulation Day '+str(self.mdg.simulationDay)
        self.resultLabel['text'] = 'Mock data stopped being simulated'

    def stopGenerating(self):
        self.mdg.continueGenerating = False
        self.continueReporting = False
        self.stopGeneratingButton.pack_forget()

    def toggleDelayParams(self, *args):
        state = self.widgetStates[self.simDelaysVar.get()]
        rows = [1, 2, 3, 4, 5]
        for r in rows:
            for w in self.delayParamsFrame.grid_slaves(row=r):
                if type(w) == tk.Frame:
                    self.changeStateFrame(state, w)
                else:
                    w['state'] = state

    def changeStateFrame(self, state, frame):
        if type(frame) == tk.Frame or type(frame) == tk.LabelFrame:
            # for w in frame.slaves()+frame.grid_slaves():
            for w in frame.winfo_children():
                if type(w) == tk.Frame or type(w) == tk.LabelFrame:
                    self.changeStateFrame(state=state, frame=w)
                else:
                    w['state'] = state
        else:
            frame['state'] = state

    def updateNumJourneys(self, *args):
        arrivalsSet = self.arrivalSetSelVar.get()
        self.arrivalSetSelNumJourneys['text'] = 'Select an arrival set'

        arrivals = pd.read_parquet(os.path.join(self.lineDataDir, 'arrivals', arrivalsSet))
        self.arrivalSetSelNumJourneys['text'] = str(arrivals.shape[0]) + ' journeys'

    def plotArrivalTimes(self):
        # Get departure time range
        # Send the range in as integers into the method - the method converts them to datetime objects
        try:
            depTimeLowInt = int(self.depTimeRangeLowEntry.get())
            depTimeHighInt = int(self.depTimeRangeHighEntry.get())
        except ValueError:
            messagebox.showinfo("Error", "Departure time range values must be integers")
            return
        
        t3 = Thread(target=lambda:self.mdg.plotArrivalTimes(arrivalTimesDir=os.path.join(self.lineDataDir, 'arrivals', 'MockRoute'), depTimeRange=(depTimeLowInt, depTimeHighInt)), daemon=True)
        t3.start()


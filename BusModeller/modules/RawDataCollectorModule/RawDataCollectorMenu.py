from tkinter import messagebox, filedialog
import tkinter as tk
from ast import literal_eval
from threading import Thread
import os
import pandas as pd
import dateutil

from configparser import ConfigParser, ExtendedInterpolation

from .RawDataCollector import RawDataCollector
# from .RawDataCollectorController import RDCController


class RawDataCollectorMenu():

    def __init__(self, menu, lineDataDir):
        self.rdc = RawDataCollector()
        # self.rdc = RDCController(lineDataDir=self.lineDataDir)
        self.menu = menu
        self.mainWindow = self.menu.window
        self.lineDataDir = lineDataDir
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        # self.config.read(os.path.join(self.lineDataDir, 'config.ini'))
        self.config.read('configs/default.ini')


    def initialize(self):

        self.borderx = 10
        self.gapx = 5
        self.gapy = 5
        

        self.mainWindow.title("RawDataCollector")

        self.welcomeLabel = tk.Label(text="Welcome to the raw data collecter\nPlease choose from below what you'd like to do", master=self.mainWindow)
        self.welcomeLabel.pack(side="top")

        #-------------------------------- Current line data folder ---------------------------------------
        self.lineDataFolderLabel = tk.Label(text="Current line data folder: "+os.path.split(self.lineDataDir)[1], master=self.mainWindow)
        self.lineDataFolderLabel.pack(side="top")
        #--------------------------------------------------------------------------------------------------

        self.contentFrame = tk.Frame(master=self.mainWindow)
        self.contentFrame.pack(side="top", fill='x')

        #---------------------- Request frame --------------------------------------------------------------
        self.requestFrame = tk.Frame(master=self.contentFrame)
        self.requestFrame.pack(side="left", fill='x')
        #----------------------------------------------------------------------------------------------


        #---------------------- Timetable data section -------------------------------------------------------------
        self.timetableFrame = tk.LabelFrame(text="Route Data Requesting", master=self.requestFrame)
        self.timetableFrame.pack(side="top", padx=self.borderx, pady=self.gapy)

        self.routeapiCallLabel = tk.Label(text="BODS URL", master=self.timetableFrame)
        self.routeapiCallLabel.grid(column=0, row=0, sticky='e')
        self.routeapiCallEntry = tk.Entry(width=30, master=self.timetableFrame)
        self.routeapiCallEntry.grid(column=1, columnspan=2, row=0, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.routeapiCallEntry.insert(index=0, string=self.config['RawDataCollector']['routedataurl'])

        self.fileLocateBool = tk.BooleanVar(value=True)
        self.fileLocationLabel = tk.Label(text="Locate file by specific file name, or by line name", master=self.timetableFrame)
        self.fileLocationLabel.grid(column=0, columnspan=2, row=1, sticky='w', padx=self.gapx)

        self.specificFileCheck = tk.Checkbutton(text="Specific File", master=self.timetableFrame, command=self.toggleSpecificFileCheck)
        self.specificFileCheck.toggle()
        self.specificFileCheck.grid(column=0, row=2, sticky='w')
        self.specificFileEntry = tk.Entry(master=self.timetableFrame, width=30)
        self.specificFileEntry.grid(column=1, row=2, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.specificFileEntry.insert(index=0, string=self.config['RawDataCollector']['specificroutefile'])
        self.specificFileEntry['state'] = 'normal'

        self.lineRefCheck = tk.Checkbutton(text="Line Name", master=self.timetableFrame, command=self.toggleLineRefCheck)
        self.lineRefCheck.grid(column=0, row=3, sticky='w')
        self.lineRefEntry = tk.Entry(master=self.timetableFrame, width=4)
        self.lineRefEntry.grid(column=1, row=3, sticky='w', padx=(self.gapx, 0), pady=self.gapy/4)
        self.lineRefEntry.insert(index=0, string=self.config['RawDataCollector']['lineref'])
        self.lineRefEntry['state'] = 'disabled'

        self.timetableButtonFrame = tk.Frame(master=self.timetableFrame)
        self.timetableButtonFrame.grid(column=0, columnspan=3, row=5)
        self.collectRouteDataButton = tk.Button(text="Collect Route Data", master=self.timetableButtonFrame, command=self.startCollectingRouteData)
        self.collectRouteDataButton.pack(side="left", pady=self.gapy)
        self.timetableProgressLabel = tk.Label(text="Progress shown here", master=self.timetableButtonFrame, state='disabled')
        self.timetableProgressLabel.pack(side="left", pady=self.gapy)
        #-------------------------------------------------------------------------------------------------------------------


        #---------------- Bus location data section ------------------------------------------------------------
        self.locationFrame = tk.LabelFrame(text="Bus Location Data Requesting", master=self.requestFrame)
        self.locationFrame.pack(side="top", padx=self.borderx, pady=self.gapy)

        self.busapiCallLabel = tk.Label(text="BODS URL", master=self.locationFrame)
        self.busapiCallLabel.grid(column=0, row=0, sticky='e', pady=self.gapy/4)
        self.busapiCallEntry = tk.Entry(master=self.locationFrame, width=30)
        self.busapiCallEntry.grid(column=1, columnspan=2, row=0, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.busapiCallEntry.insert(index=0, string=self.config['RawDataCollector']['busdataurl'])

        self.lineLabel = tk.Label(text="Line Name", master=self.locationFrame)
        self.lineLabel.grid(column=0, row=1, sticky='e', pady=self.gapy/4)
        self.lineEntry = tk.Entry(width=4, master=self.locationFrame)
        self.lineEntry.grid(column=1, row=1, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.lineEntry.insert(index=0, string=self.config['RawDataCollector']['lineref'])

        self.boundingBoxLabel = tk.Label(text="Bounding box", master=self.locationFrame)
        self.boundingBoxLabel.grid(column=0, row=2, sticky='e', pady=self.gapy/4)
        self.boundBoxBool = tk.BooleanVar()
        self.boundBoxCheck = tk.Checkbutton(variable=self.boundBoxBool, master=self.locationFrame, command=self.toggleBoundingBox)
        self.boundBoxCheck.grid(column=1, row=2, sticky='w', padx=self.gapx, pady=self.gapy/4)

        self.boundBoxLongLabel = tk.Label(text="Longitude", master=self.locationFrame, state='disabled')
        self.boundBoxLongLabel.grid(column=0, row=3, sticky='e', pady=self.gapy/4)
        self.boundBoxLongLowFrame = tk.Frame(master=self.locationFrame)
        self.boundBoxLongLowFrame.grid(column=1, row=3, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.boundBoxLongLowLabel = tk.Label(text="Low:", master=self.boundBoxLongLowFrame, state='disabled')
        self.boundBoxLongLowLabel.pack(side="left")
        self.boundBoxLongLowEntry = tk.Spinbox(from_=-180, to_=180, width=6, increment=0.1, master=self.boundBoxLongLowFrame)
        self.boundBoxLongLowEntry.pack(side="left")
        self.boundBoxLongLowEntry.delete(0, "end")
        self.boundBoxLongLowEntry.insert(index=0, s=self.config['RawDataCollector']['boundboxlonglow'])
        self.boundBoxLongLowEntry['state'] = 'disabled'
        self.boundBoxLongHighFrame = tk.Frame(master=self.locationFrame)
        self.boundBoxLongHighFrame.grid(column=2, row=3, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.boundBoxLongHighLabel = tk.Label(text="High:", master=self.boundBoxLongHighFrame, state='disabled')
        self.boundBoxLongHighLabel.pack(side="left")
        self.boundBoxLongHighEntry = tk.Spinbox(from_=-180, to_=180, width=6, increment=0.1, master=self.boundBoxLongHighFrame)
        self.boundBoxLongHighEntry.pack(side="left")
        self.boundBoxLongHighEntry.delete(0, "end")
        self.boundBoxLongHighEntry.insert(index=0, s=self.config['RawDataCollector']['boundboxlonghigh'])
        self.boundBoxLongHighEntry['state'] = 'disabled'

        self.boundBoxLatLabel = tk.Label(text="Latitude", master=self.locationFrame, state='disabled')
        self.boundBoxLatLabel.grid(column=0, row=4, sticky='e', pady=self.gapy/4)
        self.boundBoxLatLowFrame = tk.Frame(master=self.locationFrame)
        self.boundBoxLatLowFrame.grid(column=1, row=4, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.boundBoxLatLowLabel = tk.Label(text="Low:", master=self.boundBoxLatLowFrame, state='disabled')
        self.boundBoxLatLowLabel.pack(side="left")
        self.boundBoxLatLowEntry = tk.Spinbox(from_=-90, to_=90, width=6, increment=0.1, master=self.boundBoxLatLowFrame)
        self.boundBoxLatLowEntry.pack(side="left")
        self.boundBoxLatLowEntry.delete(0, "end")
        self.boundBoxLatLowEntry.insert(index=0, s=self.config['RawDataCollector']['boundboxlatlow'])
        self.boundBoxLatLowEntry['state'] = 'disabled'
        self.boundBoxLatHighFrame = tk.Frame(master=self.locationFrame)
        self.boundBoxLatHighFrame.grid(column=2, row=4, sticky='w', padx=self.gapx, pady=self.gapy/4)
        self.boundBoxLatHighLabel = tk.Label(text="High:", master=self.boundBoxLatHighFrame, state='disabled')
        self.boundBoxLatHighLabel.pack(side="left")
        self.boundBoxLatHighEntry = tk.Spinbox(from_=-90, to_=90, width=6, increment=0.1, master=self.boundBoxLatHighFrame)
        self.boundBoxLatHighEntry.pack(side="left")
        self.boundBoxLatHighEntry.delete(0, "end")
        self.boundBoxLatHighEntry.insert(index=0, s=self.config['RawDataCollector']['boundboxlathigh'])
        self.boundBoxLatHighEntry['state'] = 'disabled'
        if literal_eval(self.config['RawDataCollector']['boundbox']) == 1:
            self.boundBoxCheck.invoke()

        self.continuouslyCollectLabel = tk.Label(text="Keep Collecting", master=self.locationFrame)
        self.continuouslyCollectLabel.grid(column=0, row=8, padx=(self.gapx, 0), pady=self.gapy/4)
        self.continuouslyCollectBool = tk.BooleanVar()
        self.continuouslyCollectCheck = tk.Checkbutton(master=self.locationFrame, var=self.continuouslyCollectBool)
        if literal_eval(self.config['RawDataCollector']['continuouscollection']) == 1:
            self.continuouslyCollectCheck.invoke()
        self.continuouslyCollectCheck.grid(column=1, row=8, sticky='w', padx=self.gapx, pady=self.gapy/4)

        self.busButtonsFrame = tk.Frame(master=self.locationFrame)
        self.busButtonsFrame.grid(column=0, columnspan=3, row=9, pady=self.gapy/4)
        self.collectBusDataButton = tk.Button(text="Collect Location Data", master=self.busButtonsFrame, command=self.beginBusLocationCollection)
        self.collectBusDataButton.pack(side="left", padx=self.gapx)
        self.stopCollectingBusDataButton = tk.Button(text="Stop Collecting", master=self.busButtonsFrame, command=self.stopCollectingBusData, state='disabled')
        self.stopCollectingBusDataButton.pack(side="left", padx=self.gapx)
        self.busProgressLabel = tk.Label(text="Progress shown here", master=self.locationFrame, state='disabled')
        self.busProgressLabel.grid(column=0, columnspan=3, row=10, pady=self.gapy/4)
        #-----------------------------------------------------------------------------------------------------------------

        self.mainWindow.mainloop()



    ############################ Route Data Functions #######################################################
    def toggleSpecificFileCheck(self):
        self.fileLocateBool.set(not self.fileLocateBool.get())
        if self.fileLocateBool.get() == False:
            self.specificFileEntry['state'] = 'disabled'
            self.lineRefCheck.toggle()
            self.lineRefEntry['state'] = 'normal'
        else:
            self.specificFileEntry['state'] = 'normal'
            self.lineRefCheck.toggle()
            self.lineRefEntry['state'] = 'disabled'
    
    def toggleLineRefCheck(self):
        self.fileLocateBool.set(not self.fileLocateBool.get())
        if self.fileLocateBool.get() == False:
            self.specificFileCheck.toggle()
            self.specificFileEntry['state'] = 'disabled'
            self.lineRefEntry['state'] = 'normal'
        elif self.fileLocateBool.get() == True:
            self.specificFileCheck.toggle()
            self.specificFileEntry['state'] = 'normal'
            self.lineRefEntry['state'] = 'disabled'


    def startCollectingRouteData(self):

        entries = self.getRouteCollectionEntries()

        self.collectRouteDataButton['state'] = 'disabled'

        t1 = Thread(target=lambda:self.getRouteData(url=entries['url'], specificFile=entries['specificFile'], line=entries['line']), daemon=True)
        t1.start()

        t2 = Thread(target=self.reportRouteProgress, daemon=True)
        t2.start()

    def getRouteCollectionEntries(self):

        # Collect entries

        errors = []
        error = True

        # url ---------------------------------
        url = self.routeapiCallEntry.get()
        if len(url) == 0:
            errors.append('URL entry is empty')
            error = True
        #---------------------------------------

        # File locating ----------------------------------
        if self.fileLocateBool.get() == True:
            specificFile = self.specificFileEntry.get()
            line = None
            if len(specificFile) == 0:
                errors.append('Specific File entry is empty')
                error = True
        else:
            specificFile = None
            line = self.lineRefEntry.get()
            if len(line) == 0:
                errors.append('Line Name entry is empty')
                error = True
        #---------------------------------------------------

        return {'url':url,
                'specificFile':specificFile,
                'line':line}


    def getRouteData(self, url, specificFile, line):

        result = self.rdc.getBusRouteDataController(url=url, lineDataDir=self.lineDataDir, specificFile=specificFile, line=line)

        if result == True:
            self.keepReporting = False
            self.collectRouteDataButton['state'] = 'normal'
        # Returns 0 if there is a problem with the provided url
        elif result == 0:
            self.keepReporting = False
            messagebox.showinfo("Error", "The provided URL does not give a valid response")
            self.timetableProgressLabel['text'] = 'The provided URL does not work'
        # Return False if user provided a line name rather than a specific file and there is no file corresponding to the given line name
        elif result == 1:
            self.keepReporting = False
            messagebox.showinfo("Error", "No files from the URL response contain the given line name")
            self.timetableProgressLabel['text'] = "URL response doesn't contain line"
        # Returns 2 if there are multiple files with the same line name and user provided line name instead of specific file
        elif result == 2:
            self.keepReporting = False
            messagebox.showinfo("Error", "There are multiple stop files with the same line name.\nPlease look through them manually and find the correct one:\n"+'\n'.join(f for f in result))
            self.timetableProgressLabel['text'] = 'Multiple stop files with line name'
        # Returns 3 if user provides a specific file and the file doesn't exist
        elif result == 3:
            self.keepReporting = False
            messagebox.showinfo("Error", "Given Specific File doesn't exist")
            self.timetableProgressLabel['text'] = "File " + specificFile + " doesn't exist"
            

    def reportRouteProgress(self):
        self.keepReporting = True
        self.timetableProgressLabel['state'] = 'normal'
        n=0
        while self.keepReporting == True:
            self.timetableProgressLabel['text'] = self.rdc.progress
        self.timetableProgressLabel['state'] = 'disabled'
    ############################################################################################################


    def toggleBoundingBox(self):
        boundBoxWidgets=[self.boundBoxLongLabel, self.boundBoxLongLowLabel, self.boundBoxLongLowEntry, self.boundBoxLongHighLabel, self.boundBoxLongHighEntry, 
                 self.boundBoxLatLabel, self.boundBoxLatLowLabel, self.boundBoxLatLowEntry, self.boundBoxLatHighLabel, self.boundBoxLatHighEntry]
        if self.boundBoxBool.get() == True:
            for w in boundBoxWidgets:
                w['state'] = 'normal'
        elif self.boundBoxBool.get() == False:
            for w in boundBoxWidgets:
                w['state'] = 'disabled'

    # Collects entries from UI, evaluates if the entries are valid, then creates threads to simultaneously collect the bus data, report progress, and allow the user to stop collection
    def beginBusLocationCollection(self):

        # Collect entries

        errors = []
        error = False

        # url ------------------------------
        url = self.busapiCallEntry.get()
        #-----------------------------------

        # Line Ref ---------------------------------
        line = self.lineEntry.get()
        #-----------------------------------------

        # Bounding box ------------------------------------------------------------------------------
        boundBoxBool = self.boundBoxBool.get()
        if boundBoxBool == True:
            try:
                boundBoxLongLow = float(self.boundBoxLongLowEntry.get())
                boundBoxLongHigh = float(self.boundBoxLongHighEntry.get())
                boundBoxLatLow = float(self.boundBoxLatLowEntry.get())
                boundBoxLatHigh = float(self.boundBoxLatHighEntry.get())
                boundingBox = [[boundBoxLongLow, boundBoxLongHigh], [boundBoxLatLow, boundBoxLatHigh]]
            except ValueError:
                errors.append('Bounding box values must be real numbers')
                error = True
        else:
            boundingBox = None
        #--------------------------------------------------------------------------------------------

        # Keep collecting ---------------------------------
        keepCollecting = self.continuouslyCollectBool.get()
        #--------------------------------------------------

        if error == True:
            messagebox.showinfo('Error', 'Error with the following entries: '+str(errors))
            return

        self.collectBusDataButton['state'] = 'disabled'
        self.stopCollectingBusDataButton['state'] = 'normal'
        self.busProgressLabel['state'] = 'normal'

        t1 = Thread(target=lambda:self.collectBusLocations(url=url, 
                                                           line=line,
                                                           lineDataDir=self.lineDataDir, 
                                                           keepCollecting=keepCollecting, 
                                                           boundingBox=boundingBox
                                                           ), 
                                                        daemon=True)
        t1.start()
        t2 = Thread(target=self.reportBusProgress, daemon=True)
        t2.start()

    def collectBusLocations(self, url, line, lineDataDir, keepCollecting, boundingBox):
        self.rdc.getBusLocationData(url=url, line=line, lineDataDir=lineDataDir, keepCollecting=keepCollecting, boundingBox=boundingBox)
        self.continueReportingBus = False
        self.collectBusDataButton['state'] = 'normal'
        self.busProgressLabel['text'] = 'Collection stopped'
        self.busProgressLabel['state'] = 'disabled'

    def reportBusProgress(self):
        self.continueReportingBus = True
        while self.continueReportingBus == True:
            self.busProgressLabel['text'] = 'File '+str(self.rdc.N)
        self.busProgressLabel['text'] = 'Collection will stop shortly'

    def stopCollectingBusData(self):
        self.rdc.keepCollecting = False
        self.continueReportingBus = False
        self.stopCollectingBusDataButton['state'] = 'disabled'
    #######################################################################################################
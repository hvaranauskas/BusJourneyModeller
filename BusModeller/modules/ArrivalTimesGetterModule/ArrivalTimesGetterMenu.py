import tkinter as tk
from tkinter import messagebox
import os

from configparser import ConfigParser, ExtendedInterpolation

from .ArrivalTimesGetter import ArrivalTimesGetter


class ArrivalTimesGetterMenu():

    def __init__(self, menu, lineDataDir):
        self.atg = ArrivalTimesGetter()
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('configs/default.ini')
        self.menu = menu
        self.mainWindow = self.menu.window
        self.lineDataDir = lineDataDir


    def initialize(self):

        self.borderx = 10
        self.gapx = 5
        self.gapy = 5


        #--------------------- Initialization ---------------------------------------------------
        self.mainWindow.title('ArrivalTimesGetter')
        #-------------------------------------------------------------------------------------------

        #--------------------- Welcome ---------------------------------------------------------------
        self.welcomeLabel = tk.Label(text="Welcome to the Arrival Times Getter!", master=self.mainWindow)
        self.welcomeLabel.pack(side="top", padx=self.borderx, pady=self.gapy)
        #-------------------------------------------------------------------------------------------

        #----------------------------- Current line data folder ------------------------------------------
        self.lineDataFolderLabel = tk.Label(text="Current line data folder: "+os.path.split(self.lineDataDir)[1], master=self.mainWindow)
        self.lineDataFolderLabel.pack(side="top", padx=self.borderx, pady=self.gapy)
        #-------------------------------------------------------------------------------------------------


        #------------------------- Intersection RLs -----------------------------------------------------------------------------------------
        # self.intersectionRLsFrame = tk.LabelFrame(text="Compute intersections RLs", master=self.mainWindow)
        # self.intersectionRLsFrame.pack(side="top", fill='x', padx=self.borderx, pady=self.gapy)
        # self.intersectionRouteDirLabel = tk.Label(text="Route DF Directory", master=self.intersectionRLsFrame)
        # self.intersectionRouteDirLabel.grid(column=0, row=0)
        # self.intersectionRouteDirEntry = tk.Entry(width=30, master=self.intersectionRLsFrame)
        # self.intersectionRouteDirEntry.grid(column=1, row=0)
        # self.intersectionRouteDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['intersectionroutedir'])
        # self.intersectionRouteLinkDirLabel = tk.Label(text="Route Link DF Directory", master=self.intersectionRLsFrame)
        # self.intersectionRouteLinkDirLabel.grid(column=0, row=1)
        # self.intersectionRouteLinkDirEntry = tk.Entry(width=30, master=self.intersectionRLsFrame)
        # self.intersectionRouteLinkDirEntry.grid(column=1, row=1)
        # self.intersectionRouteLinkDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['intersectionroutelinkdir'])
        # self.intersectionStorageDirLabel = tk.Label(text="Intersection RL Storage Directory", master=self.intersectionRLsFrame)
        # self.intersectionStorageDirLabel.grid(column=0, row=2)
        # self.intersectionStorageDirEntry = tk.Entry(width=30, master=self.intersectionRLsFrame)
        # self.intersectionStorageDirEntry.grid(column=1, row=2)
        # self.intersectionStorageDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['intersectionstoragedir'])
        # self.computeIntersectionsButton = tk.Button(text="Compute intersection RLs", command=self.computeIntersectionRLs, master=self.intersectionRLsFrame)
        # self.computeIntersectionsButton.grid(column=0, columnspan=2, row=3)

        self.intersectionRLsLabel = tk.Label(text="You need to compute the intersection RLs before you can compute arrival times", master=self.mainWindow)
        self.intersectionRLsLabel.pack(side="top")

        self.computeInersectionsButton = tk.Button(text="Compute intersection RLs", command=self.computeIntersectionRLs, master=self.mainWindow)
        self.computeInersectionsButton.pack(side="top", padx=self.borderx, pady=self.gapy)
        #----------------------------------------------------------------------------------------------------------------------------------------

        if 'IntersectionRLs.csv' in os.listdir(self.lineDataDir):
            self.buildArrivalsFrame()

        self.mainWindow.mainloop()


    def computeIntersectionRLs(self):
        self.atg.computeIntersectionsRLs(self.lineDataDir)
        self.buildArrivalsFrame()


    def buildArrivalsFrame(self):

        try:
            if self.builtArrivalsFrame == True:
                return
        except AttributeError:
            pass

        #------------------------------------- Arrival Times ----------------------------------------------------------------------------
        self.arrivalsFrame = tk.LabelFrame(text="Arrival times", master=self.mainWindow)
        self.arrivalsFrame.pack(side="top", fill='x', padx=self.borderx, pady=self.gapy)

        # self.busLocationDirLabel = tk.Label(text="Bus location DF dir", master=self.arrivalsFrame)
        # self.busLocationDirLabel.grid(column=0, row=0)
        # self.busLocationDirEntry = tk.Entry(width=30, master=self.arrivalsFrame)
        # self.busLocationDirEntry.grid(column=1, row=0)
        # self.busLocationDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsbuslocationdir'])
        # self.routeDFDirLabel = tk.Label(text="Route DF dir", master=self.arrivalsFrame)
        # self.routeDFDirLabel.grid(column=0, row=1)
        # self.routeDFDirEntry = tk.Entry(width=30, master=self.arrivalsFrame)
        # self.routeDFDirEntry.grid(column=1, row=1)
        # self.routeDFDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsroutedir'])
        # self.routeLinkDFDirLabel = tk.Label(text="RouteLink DF Directory", master=self.arrivalsFrame)
        # self.routeLinkDFDirLabel.grid(column=0, row=2)
        # self.routeLinkDFDirEntry = tk.Entry(master=self.arrivalsFrame)
        # self.routeLinkDFDirEntry.grid(column=1, row=2)
        # self.routeLinkDFDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsroutelinkdir'])
        # self.journeyCodesDFDirLabel = tk.Label(text="Journey Codes DF Directory", master=self.arrivalsFrame)
        # self.journeyCodesDFDirLabel.grid(column=0, row=3)
        # self.journeyCodesDFDirEntry = tk.Entry(master=self.arrivalsFrame)
        # self.journeyCodesDFDirEntry.grid(column=1, row=3)
        # self.journeyCodesDFDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsjourneycodesdir'])
        # self.intersectionRLsDirLabel = tk.Label(text="Intersection RLs Directory", master=self.arrivalsFrame)
        # self.intersectionRLsDirLabel.grid(column=0, row=4)
        # self.intersectionRLsDirEntry = tk.Entry(master=self.arrivalsFrame)
        # self.intersectionRLsDirEntry.grid(column=1, row=4)
        # self.intersectionRLsDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsintersectionrlsdir'])

        self.discontThreshLabel = tk.Label(text="Discontinuity threshold (minutes)", master=self.arrivalsFrame)
        self.discontThreshLabel.grid(column=0, row=5)
        self.discontThreshEntry = tk.Spinbox(from_=0, to_=20, increment=1, master=self.arrivalsFrame)
        self.discontThreshEntry.grid(column=1, row=5)
        self.discontThreshEntry.delete(0,"end")
        self.discontThreshEntry.insert(index=0, s=self.config['ArrivalTimesGetter']['discontthresh'])
        self.arrivalRadiusLabel = tk.Label(text="Arrival Radius", master=self.arrivalsFrame)
        self.arrivalRadiusLabel.grid(column=0, row=6)
        self.arrivalRadiusEntry = tk.Spinbox(from_=0, to_=0.01, increment=0.0001, master=self.arrivalsFrame)
        self.arrivalRadiusEntry.grid(column=1, row=6)
        self.arrivalRadiusEntry.delete(0,"end")
        self.arrivalRadiusEntry.insert(index=0, s=self.config['ArrivalTimesGetter']['arrivalradius'])
        self.diffRouteThreshLabel = tk.Label(text="Different route threshold", master=self.arrivalsFrame)
        self.diffRouteThreshLabel.grid(column=0, row=7)
        self.diffRouteThreshEntry = tk.Spinbox(from_=0, to_=1, increment=0.0001, master=self.arrivalsFrame)
        self.diffRouteThreshEntry.grid(column=1, row=7)
        self.diffRouteThreshEntry.delete(0,"end")
        self.diffRouteThreshEntry.insert(index=0, s=self.config['ArrivalTimesGetter']['diffroutethresh'])
        # self.arrivalsDFDirLabel = tk.Label(text="Arrivals DF Storage Dir", master=self.arrivalsFrame)
        # self.arrivalsDFDirLabel.grid(column=0, row=8)
        # self.arrivalsDFDirEntry = tk.Entry(width=30, master=self.arrivalsFrame)
        # self.arrivalsDFDirEntry.grid(column=1, row=8)
        # self.arrivalsDFDirEntry.insert(index=0, string=self.config['ArrivalTimesGetter']['arrivalsstoragedir'])
        self.arrivalsButtonFrame = tk.Frame(master=self.arrivalsFrame)
        self.arrivalsButtonFrame.grid(column=0, columnspan=2, row=9)
        self.computeArrivalsButton = tk.Button(text="Compute arrivals", master=self.arrivalsButtonFrame, command=self.computeArrivals)
        self.computeArrivalsButton.pack(side="left")
        #--------------------------------------------------------------------------------------------------------------------------

        self.builtArrivalsFrame = True



    def computeArrivals(self):
        timeGapDiscontThresh = self.discontThreshEntry.get()
        arrivalRadius = float(self.arrivalRadiusEntry.get())
        diffRouteThreshold = float(self.diffRouteThreshEntry.get())
        
        self.atg.computeArrivals(lineDataDir = self.lineDataDir,
                                 timeGapDiscontThresh=timeGapDiscontThresh,
                                 arrivalRadius=arrivalRadius,
                                 diffRouteThreshold=diffRouteThreshold,
                                 )
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import shutil
from configparser import ConfigParser, ExtendedInterpolation

from .MenuNode import MenuNode

# Basic menu entry-point to program to display different modules to user
class MainMenu():

    def __init__(self):
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.read('configs/default.ini')

    def initializeMenu(self):

        self.borderx = 10
        self.gapx = 5
        self.gapy = 5

        #------------------------------------ Labels --------------------------------------------------------
        self.mainMenu = MenuNode()
        self.mainMenu.window.title("Bus Modeller - Main Menu")

        self.welcomeLabel = ttk.Label(text="Main menu", master=self.mainMenu.window)
        self.welcomeLabel.pack(padx=self.borderx, pady=(10,1))
        #------------------------------------------------------------------------------------------------------

        #--------------------------------------- Line data directory selection ---------------------------------------------
        self.lineDataDirFrame = ttk.Labelframe(text="Line Data Folders")
        self.lineDataDirFrame.pack(side="top", padx=self.borderx, pady=self.gapy)
        self.lineDataDirLabel = ttk.Label(text="Bus Data is stored across multiple DataFrames\nDataFrames for a line are kept together in one folder\nPlease select/create such a folder below to use the modules", master=self.lineDataDirFrame)
        self.lineDataDirLabel.pack(side="top", padx=self.gapx)

        self.createLineDataDirFrame = ttk.Frame(master=self.lineDataDirFrame)
        self.createLineDataDirFrame.pack(side="top", padx=self.gapx, pady=(0, self.gapy))
        self.createDirLabel = ttk.Label(text="Create folder", master=self.createLineDataDirFrame)
        self.createDirLabel.pack(side="top", anchor='w')
        self.createLineDataDirEntry = ttk.Entry(width=20, master=self.createLineDataDirFrame)
        self.createLineDataDirEntry.pack(side="left", fill='x', anchor="e")
        self.createLineDataDirEntry.insert(index=0, string=self.config['General']['defaultlinedata'])
        self.createDirButton = ttk.Button(text="Create folder", width=12, master=self.createLineDataDirFrame, command=self.createLineFolder)
        self.createDirButton.pack(side="right", fill='x', anchor="w")

        self.selectLineDataDirFrame = ttk.Frame(master=self.lineDataDirFrame)
        self.selectLineDataDirFrame.pack(side="top", padx=self.gapx, pady=(0, self.gapy))
        self.selectDirLabel = ttk.Label(text="Select directory", master=self.selectLineDataDirFrame)
        self.selectDirLabel.pack(side="top", anchor='w')
        self.selectionLineDataVar = tk.StringVar()
        self.selectionLineDataVar.trace('w', self.updateDefaultLineDataDir)

        self.availableLineDataDirs = os.listdir(self.config['General']['linedatadir'])
        self.selectLineDataDirEntry = ttk.Combobox(width=30, values=self.availableLineDataDirs, master=self.selectLineDataDirFrame, state='readonly', textvariable=self.selectionLineDataVar)
        self.selectLineDataDirEntry.pack(side="left", fill='x')
        #----------------------------------------------------------------------------------------------------------------------------------

        #--------------------------------------------- Module buttons --------------------------------------------------
        self.rdcButton = ttk.Button(text="Raw Data Collector", command=self.startRDC, master=self.mainMenu.window, state='disabled')
        self.rdcButton.pack(fill='x', padx=self.borderx, pady=1)

        self.atgButton = ttk.Button(text="Arrival Times Getter", command=self.startATG, master=self.mainMenu.window, state='disabled')
        self.atgButton.pack(fill='x', padx=self.borderx, pady=1)

        self.mdgButton = ttk.Button(text="Mock Data Generator", command=self.startMDG, master=self.mainMenu.window, state='disabled')
        self.mdgButton.pack(fill='x', padx=self.borderx, pady=1)

        self.rwmButton = ttk.Button(text="Random Walk Modeller", command=self.startRWM, master=self.mainMenu.window, state='disabled')
        self.rwmButton.pack(fill='x', padx=self.borderx, pady=(1,10))
        #---------------------------------------------------------------------------------------------------------------

        # Entry inserted here since when an entry is added, self.updateDefaultLineDataDir is called, which interacts with the module buttons. Throws an error if done before the buttons
        if os.path.exists(os.path.join(self.config['General']['linedatadir'], self.config['General']['defaultlinedataselection'])):
            self.selectLineDataDirEntry.set(value=self.config['General']['defaultlinedataselection'])

        # Only ever used when program is first launched so that user doesn't try to run a module without choosing a line data folder
        if len(self.selectionLineDataVar.get()) != 0:
            self.enableButtons()

        self.mainMenu.window.mainloop()



    def createLineFolder(self):
        folderName = self.createLineDataDirEntry.get()
        dir = os.path.join(self.config['General']['linedatadir'], folderName)
        if os.path.exists(dir):
            if messagebox.askokcancel("Folder already exists", "The folder "+folderName+" already exists.\nIf you continue, it will be overwritten.\nContinue?") == False:
                return
            else:
                shutil.rmtree(dir)

        os.mkdir(dir)
        # shutil.copyfile(src='configs/default.ini', dst=os.path.join(dir, 'default.ini'))
        self.availableLineDataDirs = os.listdir(self.config['General']['linedatadir'])
        self.selectLineDataDirEntry['values'] = self.availableLineDataDirs
        self.selectLineDataDirEntry.set(folderName)

        self.updateDefaultLineDataDir()

    def updateDefaultLineDataDir(self, *args):
        self.availableLineDataDirs = os.listdir(self.config['General']['linedatadir'])
        self.config['General']['defaultlinedataselection'] = self.selectionLineDataVar.get()
        self.config.write(open('configs/default.ini', 'w'))

        if len(self.selectionLineDataVar.get()) != 0:
            self.enableButtons()

    def enableButtons(self):
        self.rdcButton['state'] = 'normal'
        self.atgButton['state'] = 'normal'
        self.mdgButton['state'] = 'normal'
        self.rwmButton['state'] = 'normal'


    # Initialize RawDataCollector module
    def startRDC(self):

        # Don't open duplicate windows
        for c in self.mainMenu.children:
            if c.window.title() == 'RawDataCollector':
                messagebox.showinfo('Instance already running', 'You already have a RawDataCollector module open.\nYou can only have one open at a time.')
                return
            
        from . import RawDataCollectorModule as rdc

        rdcMenu = MenuNode(parent=self.mainMenu)
        self.rdcModule = rdc.RawDataCollectorMenu(menu=rdcMenu, lineDataDir=os.path.join(self.config['General']['linedatadir'], self.selectionLineDataVar.get()))
        self.rdcModule.initialize()

    # Initialize ArrivalTimeGetter module
    def startATG(self):

        # Don't open duplicate windows
        for c in self.mainMenu.children:
            if c.window.title() == 'ArrivalTimesGetter':
                messagebox.showinfo('Instance already running', 'You already have an ArrivalTimesGetter module open.\nYou can only have one open at a time.')
                return
            
        errors = []
        lineDataDir = os.path.join(self.config['General']['linedatadir'], self.selectionLineDataVar.get())
        files = os.listdir(lineDataDir)
        if 'Routes.csv' not in files:
            errors.append('Routes.csv')
        if 'RouteLinks.parquet' not in files:
            errors.append('RouteLinks.parquet')
        if 'RawBusLocations.parquet' not in files:
            errors.append('RawBusLocations.parquet')
        if len(errors) != 0:
            messagebox.showinfo('Missing files', 'You need to compute the following files in the Raw Data Collector to get arrival times:\n'+'\n'.join(str(e) for e in errors))
            return

        from . import ArrivalTimesGetterModule as atg

        atgMenu = MenuNode(parent=self.mainMenu)
        self.atgModule = atg.ArrivalTimesGetterMenu(menu=atgMenu, lineDataDir=lineDataDir)
        self.atgModule.initialize()

    # Initialize MockDataGenerator module
    def startMDG(self):

        # Don't open duplicate windows
        for c in self.mainMenu.children:
            if c.window.title() == 'MockDataGenerator':
                messagebox.showinfo('Instance already running', 'You already have a MockDataGenerator module open.\nYou can only have one open at a time.')
                return
        
        # Ensure that the directory has no real-world data stored in it
        error = False
        lineDataDir = os.path.join(self.config['General']['linedatadir'], self.selectionLineDataVar.get())
        files = os.listdir(lineDataDir)
        if 'Routes.csv' in files or 'RouteLinks.parquet' in files or 'RawBusLocations.parquet' in files:
            error = True
        if error == True:
            messagebox.showinfo('Occupied Directory', 'This line-data folder has real-world data in it.\nPlease select/create a folder without real-world data in it.')
            return
        
        # Imports for this module can take long, so only do it if user starts the module
        from . import MockDataGeneratorModule as mdg

        # Initializes the module
        mdgMenu = MenuNode(parent=self.mainMenu)
        self.mdgModule = mdg.MockDataGeneratorMenu(menu=mdgMenu, lineDataDir=lineDataDir)
        self.mdgModule.initialize()

    # Initializes RandomWalkModeller module
    def startRWM(self):
        
        # Don't open duplicate windows
        for c in self.mainMenu.children:
            if c.window.title() == 'RandomWalkModeller':
                messagebox.showinfo('Instance already running', 'You already have a RandomWalkModeller module open.\nYou can only have one open at a time.')
                return
        
        errors = []
        lineDataDir = os.path.join(self.config['General']['linedatadir'], self.selectionLineDataVar.get())
        files = os.listdir(lineDataDir)
        if 'arrivals' not in files:
            errors.append('arrivals')
        if len(errors) != 0:
            messagebox.showinfo('Missing files', 'You need to get the following files in the Arrival Time Getter or Mock Data Generator to compute a model:\n'+'\n'.join(str(e) for e in errors))
            return

        # Imports for this module can take long, so only do it if user starts the module
        from . import RandomWalkModellerModule as rwm

        # Initializes the module
        rwmMenu = MenuNode(parent=self.mainMenu)
        self.rwmModule = rwm.RandomWalkModellerMenu(menu=rwmMenu, lineDataDir=lineDataDir)
        self.rwmModule.initialize()
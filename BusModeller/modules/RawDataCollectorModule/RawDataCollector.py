import requests
import json
import lxml.etree as ET
from os.path import exists, join
import time
from threading import Thread
import pandas as pd
import zipfile
from io import BytesIO
import datetime


class RawDataCollector():

    def __init__(self):
        self.finishedWaiting = True
        self.progress = ''


    ####################################### Route data functions #######################################################

    # Calls getBusRouteData and stores the returned DataFrames in the given lineDataDir
    # Acts as an intermediate between the front end and the getBusRouteData function
    def getBusRouteDataController(self, url, lineDataDir, specificFile=None, line=None):
        response = self.getBusRouteData(url=url, specificFile=specificFile, line=line)
        if type(response) == int:
            return response

        fileText, journeyCodeDF, routeDF, routeLinkDF = response
        
        with open(join(lineDataDir, 'RawTimetableData.xml'), 'w') as f:
            f.write(fileText)

        journeyCodeDF.to_csv(join(lineDataDir, 'JourneyCodes.csv'))
        routeDF.to_csv(join(lineDataDir, 'Routes.csv'))
        routeLinkDF.to_parquet(join(lineDataDir, 'RouteLinks.parquet'))

        return True

    # Extracts route data corresponding to the user-given bus line into pandas DataFrames and then returns these DataFrames
    # Data is provided by the BODS API
    # Sometimes, the API can give an invalid response, in which case, the function does not extract any data and lets the user know
    def getBusRouteData(self, url, specificFile=None, line=None):

        #---------------- Get the archive of data from the BODS ---------------------------------
        self.progress = "Requesting data..."
        response = self.__requestRouteData(url)
        # If url is invalid, end function and report situation
        if type(response)==int:
            return 0
        else:
            response, lines = response
        #------------------------------------------------------------------------------
        
        #------- Try to find specificFile using line, if specificFile isn't given --------------------------------------------------
        # If a line is given as opposed to a specific file name, try to find the file corresponding to the given line
        if line != None:
            # 'lines' is a list containing all the line names referenced in the response files, so if 'line' is not in 'lines', there won't be a specificFile containing it
            if line not in lines:
                return 1
            self.progress = "Data received\nLocating stop file..."
            files = self.__locateFileByLine(response, line)

            # If there are no files or multiple files with the given line, report situation and end function
            if len(files) == 0:
                return 1
            elif len(files) > 1:
                return 2
            else:
                specificFile = files[0]
        #------------------------------------------------------------------------------------------------------------------------
        
        #----------------- Extract route data from the specificFile --------------------------------------------
        self.progress = "Stop file located\nExtracting data from file..."

        # KeyError occurs when 'specificFile' is not in the zip file returned by the BODS
        try:
            fileText = response.read(specificFile)
        except KeyError:
            return 3
        
        journeyCodeDF, routeDF, routeLinkDF = self.__extractRouteData(fileText=fileText)
        #------------------------------------------------------------------------------------------------------
        
        return fileText, journeyCodeDF, routeDF, routeLinkDF

    # Returns a zipfile object obtained from calling url
    # zipfile contains multiple 'timetable' files, each of which contain bus route data about one or more bus lines
    def __requestRouteData(self, url):

        try:
            response=requests.get(url)
        # Catches response exceptions, which are often caused by an invalid url
        # When this happens, report situation and end function
        except requests.exceptions.RequestException as e:
            return 0

        # Initial response is given as a json object with many values
        # Only value of interest is the url, which gets the TransXChange files
        j = json.loads(response.text)
        url = j['url']

        response = requests.get(url)

        self.progress = "Response got..."

        # Response is given as a zip file, so use the zipfile library to store the response
        return zipfile.ZipFile(BytesIO(response.content), 'r'), j['lines']

    def __locateFileByLine(self, response, line):

        files = []
        # Iterate through each file in the archive
        for f in response.namelist():

            dataTree = ET.fromstring(response.read(f))
            lineName = dataTree.find('.//{http://www.transxchange.org.uk/}LineName[.="'+line+'"]')
            if lineName != None:
                # If the given line is in file f, this file corresponds to a file for a bus service with line name 'line'
                files.append(f)

        return files

    # Extracts data from 'specificFile' in 'response' given by '__requestRouteData' into three DataFrames: journeyCodeDF, routeDF and routeLinkDF
    # journeyCodeDF maps each available JourneyCode onto a single RouteRef and a single departure time
    # routeDF maps each route ID onto a sequence of route link IDs
    # routeLinkDF maps each route link ID onto a sequence of longitudinal-latitudinal co-ordinates
    def __extractRouteData(self, fileText:str):
        
        # Initialize route data DFs
        journeyCodeDF = pd.DataFrame(columns=['RouteRef', 'DepTime'], dtype='object')
        routeDF = pd.DataFrame(columns=['Direction'], dtype='object') # Additional 'route link' columns are added to this DF as necessary
        routeLinkDF = pd.DataFrame(columns=['From', 'To'], dtype='object') # Additional 'location' columns are added to this DF as necessary

        dataTree = ET.fromstring(fileText)

        # Get routes by JourneyCode
        # This allows for mapping of live bus locations onto routes through their 'DatedVehicleJourneyRef'
        journeyCodes = dataTree.findall('.//{http://www.transxchange.org.uk/}JourneyCode')
        for jc in journeyCodes:

            depTime = jc.find('../../..//{http://www.transxchange.org.uk/}DepartureTime').text

            # The VehicleJourney which JourneyCode belongs to contains a JourneyPatternRef
            jpRef = jc.find('../../..//{http://www.transxchange.org.uk/}JourneyPatternRef').text
            jP = dataTree.find('.//{http://www.transxchange.org.uk/}JourneyPattern[@id="'+jpRef+'"]')
            # The corresponding JourneyPattern contains a RouteRef
            routeRef = jP.find('{http://www.transxchange.org.uk/}RouteRef').text

            # Record JourneyCode-RouteRef connections in journeyCodeDF
            journeyCodeDF.loc[jc.text] = [routeRef, depTime]

            # If info about route with reference routeRef is already in the dataframe, don't put data into dataframe again
            # Multiple journey codes can link to the same route, so this prevents the route from being extracted again for repeat references
            if routeRef in routeDF.index:
                continue
            
            # Identify direction of route
            direction = jP.find('{http://www.transxchange.org.uk/}Direction').text

            #Initialize row in dataframe for routeRef
            routeDF.loc[routeRef] = [direction] + [None]*(routeDF.shape[1]-1)

            # Now extract information about routes
            route = dataTree.find('.//{http://www.transxchange.org.uk/}Route[@id="'+routeRef+'"]')
            #Each Route has a reference to one RouteSectionRef
            routeSectionRef = route.find('.//{http://www.transxchange.org.uk/}RouteSectionRef').text
            routeSection = dataTree.find('.//{http://www.transxchange.org.uk/}RouteSection[@id="'+routeSectionRef+'"]')

            routeLinkCounter = 0
            # A RouteSection is comprised of a number of RouteLinks
            # Extract details about each routeLink in routeSection
            for routeLink in routeSection:

                # Each route in routeDF is described as a sequence of RouteLinks, which are added here
                # Extend columns in routeDF incrementally when needed
                if 'RouteLink' + str(routeLinkCounter) not in routeDF:
                    routeDF['RouteLink' + str(routeLinkCounter)] = pd.NA
                # Add routeLink's id to the route that it belongs to
                routeDF['RouteLink' + str(routeLinkCounter)][routeRef] = routeLink.attrib['id']


                # Record the stops at each end of the routeLink (The 'From' stop and the 'To' stop)
                originStopRef = routeLink.find('.//{http://www.transxchange.org.uk/}From/{http://www.transxchange.org.uk/}StopPointRef').text
                destStopRef = routeLink.find('.//{http://www.transxchange.org.uk/}To/{http://www.transxchange.org.uk/}StopPointRef').text
                
                # Initialize row for routeLink in routeLinkDF
                if routeLink.attrib['id'] not in routeLinkDF.index:
                    routeLinkDF.loc[routeLink.attrib['id']] = [originStopRef, destStopRef] + [None]*(routeLinkDF.shape[1]-2)

                # Route links are made up of many locations, which, when connected together describe the structure of the route link which they belong to
                locations = routeLink.findall('.//{http://www.transxchange.org.uk/}Location')

                # If routeLink is the first route link in the route 'route', include the location of the 'From' stop of the route link
                # The 'From' stop is not included for every other route link
                # This helps in arrival times calculations later on
                if routeLinkCounter == 0:
                    s=0
                else:
                    s=1

                routeLinkRef = routeLink.attrib['id']
                for m in range(s, len(locations)):
                    # Extend columns in routelinkDF if needed
                    routeLinkLength = routeLinkDF.loc[routeLinkRef].dropna().shape[0]
                    index = 'Location' + str(routeLinkLength-2)
                    if routeLinkLength == routeLinkDF.shape[1]:
                        routeLinkDF[index] = pd.NA
                    longitude = float(locations[m].find('.//{http://www.transxchange.org.uk/}Longitude').text)
                    latitude = float(locations[m].find('.//{http://www.transxchange.org.uk/}Latitude').text)
                    routeLinkDF.at[routeLinkRef, index] = (longitude, latitude)

                routeLinkCounter += 1
            
        # journeyCodeDF.to_csv(join(lineDataDir, 'JourneyCodes.csv'))
        # routeDF.to_csv(join(lineDataDir, 'Routes.csv'))
        # routeLinkDF.to_parquet(join(lineDataDir, 'RouteLinks.parquet'))

        return journeyCodeDF, routeDF, routeLinkDF
    
    ################################################################################################################




    ##################################### Bus data functions ##########################################################

    # Function called from front-end which acts as intermediate between the front-end and getBusLocationData
    # Calls getBusLocationData for however long user specifies and then stores the resulting DataFrame
    def getBusLocationDataController(self, url, line, lineDataDir, keepCollecting, boundingBox):
        if exists(join(lineDataDir, 'RawBusLocations.parquet')):
            rawLocationDF = pd.read_parquet(join(lineDataDir, 'RawBusLocations.parquet'))
        else:
            rawLocationDF = None
        
        self.N = 0
        self.keepCollecting = keepCollecting
        while True:

            # Used to space out API calls by 10 seconds
            startTime = datetime.datetime.now()

            response = self.getBusLocationData(url=url, line=line, rawLocationDF=rawLocationDF, boundingBox=boundingBox)

            # response is bool type when the requested URL returns an invalid response
            if type(response) == bool:
                self.keepCollecting = False
            else:
                rawLocationDF = response

            # self.keepCollecting is set to False from frontend. Doing this stops the program from requesting and extracting data
            if self.keepCollecting == False:
                break

            self.N += 1

            # Stop the program until it's been 10 seconds since the last API call
            timeLeft = datetime.timedelta(seconds=10) - (datetime.datetime.now() - startTime)
            if timeLeft >= datetime.timedelta(seconds=0):
                time.sleep(timeLeft)

        rawLocationDF.to_parquet(join(lineDataDir, 'RawBusLocations.parquet'))
        return True


    # Calls the API for the given url response, parses the response using lxml.etree, then stores the location and route data of the bus in a DataFrame
    def getBusLocationData(self, url, line, rawLocationDF=None, boundingBox=None):

        # Initialize a DataFrame if one isn't given
        if type(rawLocationDF) != pd.DataFrame:
            rawLocationDF = pd.DataFrame(dtype='object')

        # Getting response from bus location API is easier than it is for the route API, since it directly gives the XML file as the response
        response = requests.get(url)

        dataTree = ET.fromstring(response.text)

        try:
            responseTimestamp = dataTree.find('.//{http://www.siri.org.uk/siri}ResponseTimestamp').text
        # Gives attribute error when requested URL returns an invalid response
        except AttributeError:
            self.loop = False
            return False
        
        # Each VehicleActivity corresponds to a real-world bus
        # Relevant VehicleActivitys are found by their 'LineRef' element
        vehicleActivities = dataTree.findall('.//{http://www.siri.org.uk/siri}LineRef[.="'+line+'"]/../..')
        for v in vehicleActivities:

            # Get the longitude and latitude of the VehicleActivity element
            longitude = float(v.find('.//{http://www.siri.org.uk/siri}Longitude').text)
            latitude = float(v.find('.//{http://www.siri.org.uk/siri}Latitude').text)

            # Locations can often be (0,0), which is likely an error - these aren't useful
            # Also ensure location is in bounding box if one is given
            if (longitude != 0 and latitude != 0) and (boundingBox == None or (longitude > boundingBox[0][0] and longitude < boundingBox[1][0] and latitude < boundingBox[0][1] and latitude > boundingBox[1][1])):

                # Location of a bus at the time at which the file was received is put into a column with column name as the bus' unique reference ID concatenated with a reference to the journey code it's travelling along
                vehRef = v.find('.//{http://www.siri.org.uk/siri}VehicleRef').text
                journeyCode = v.find('.//{http://www.siri.org.uk/siri}DatedVehicleJourneyRef').text
                index = vehRef + ' ' + journeyCode

                # Add new column corresponding to the vehicle on the journey if there isn't already one in the DF
                if index not in rawLocationDF:
                    rawLocationDF = pd.concat(objs=[rawLocationDF, pd.Series(name=index, index=rawLocationDF.index, data=[None]*rawLocationDF.shape[0], dtype='object')], axis=1)
                # Add new row corresponding to the time at which the file was received
                if responseTimestamp not in rawLocationDF.index:
                    rawLocationDF.loc[responseTimestamp] = pd.Series([None]*rawLocationDF.shape[1], dtype='object')
                # Add location to relevant cell
                # Cells in a parquet file have strict type requirements - have to convert location values to strings to store DF as a parquet
                rawLocationDF[index][responseTimestamp] = (str(longitude), str(latitude))

        return rawLocationDF

    ####################################################################################################################
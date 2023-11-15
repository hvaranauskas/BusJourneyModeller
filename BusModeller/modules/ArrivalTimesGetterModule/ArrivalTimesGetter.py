import os
import pandas as pd
import pickle
import gzip
import lxml.etree as ET
import dateutil
import datetime
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt


class ArrivalTimesGetter():

    ########################### Intersection RL methods ######################################################################

    def computeIntersectionsRLs(self, lineDataDir):

        self.fig = plt.gcf()
        self.ax = self.fig.gca()

        routeDF = pd.read_csv(os.path.join(lineDataDir, 'Routes.csv'), index_col=0).astype('object')
        routeLinkDF = pd.read_parquet(os.path.join(lineDataDir, 'RouteLinks.parquet')).astype('object')
        intersectionRLs = pd.DataFrame(columns=['IntersectionRL1'], dtype='object')

        for r in routeDF.index:

            print(routeDF.index.get_loc(r), routeDF.shape[0])
            
            if r not in intersectionRLs.index:
                intersectionRLs.loc[r] = pd.Series([None]*intersectionRLs.shape[1], dtype='string')

            # Compare each routelink in the route with each other routelink in the route
            for rl in routeDF.loc[r, 'RouteLink0':].dropna():

                # print(routeDF.loc[r].tolist().index(rl), routeDF.loc[r].dropna().shape[0])

                if rl in intersectionRLs.loc[r].dropna().tolist():
                    continue

                result = self.computeIfIntersectionRL(rl, r, routeDF, routeLinkDF)

                if result == None:
                    continue
                
                # Add intersection RLs to the DF
                if result not in intersectionRLs.loc[r].dropna().values.tolist():
                    while intersectionRLs.shape[1] - intersectionRLs.loc[r].dropna().shape[0] < 2:
                        intersectionRLs['IntersectionRL'+str(intersectionRLs.shape[1]+1)] = None
                    intersectionRLs.at[r, 'IntersectionRL'+str(intersectionRLs.loc[r].dropna().shape[0]+1)] = rl
                    intersectionRLs.at[r, 'IntersectionRL'+str(intersectionRLs.loc[r].dropna().shape[0]+1)] = result
                
                else:
                    while intersectionRLs.shape[1] - intersectionRLs.loc[r].dropna().shape[0] < 1:
                        intersectionRLs['IntersectionRL'+str(intersectionRLs.shape[1]+1)] = None
                    intersectionRLs.at[r, 'IntersectionRL'+str(intersectionRLs.loc[r].dropna().shape[0]+1)] = rl
            
        intersectionRLs.to_csv(os.path.join(lineDataDir, 'IntersectionRLs.csv'))

        
    def computeIfIntersectionRL(self, rl, route, routeDF, routeLinkDF):

            routeLink = routeLinkDF.loc[rl, 'Location0':].dropna()

            # Compare routeLink to every other route link in the route
            rlLabel = routeDF.loc[route][routeDF.loc[route]==rl].index[0]
            # for otherRL in routeDF.loc[route, 'RouteLink0':].dropna():
            for otherRL in routeDF.loc[route, rlLabel:].dropna():

                # Ensure otherRL is not the same routelink as rl - not doing this will mark every routelink as an intersection routelink
                if otherRL == rl:
                    continue

                otherRouteLink = routeLinkDF.loc[otherRL, 'Location0':].dropna()

                # Compare each pair of locations in rl with each pair of locations in otherRL
                for l in range(1, routeLink.shape[0]-2):
                    
                    if min(np.linalg.norm(routeLink[l] - otherRouteLink[m]) for m in range(otherRouteLink.shape[0])) > 0.001:
                        continue

                    for otherL in range(1, otherRouteLink.shape[0]-2):
                    # for otherL in range(1, shorterDists.shape[0]-2):

                        # if shorterDists[otherL] > adjDist and shorterDists[otherL+1] > adjDist:
                        # if shorterDists[otherL] and shorterDists[otherL+1]
                        #     print("Skip")
                        #     continue
                        # print("No skip")

                        # self.ax.cla()
                        # # for rRef in routeDF:
                        # # for rlRef in routeDF.loc[route, 'RouteLink0':].dropna():
                        # locations = routeLinkDF.loc[rl, 'Location0':].dropna()
                        # for m in range(locations.shape[0]-1):
                        #     if m == l or m == otherL:
                        #         # self.ax.plot([float(locations[m][0]), float(locations[m+1][0])], [float(locations[m][1]), float(locations[m+1][1])], 'v', alpha=1)
                        #         self.ax.plot([float(locations[m][0]), float(locations[m+1][0])], [float(locations[m][1]), float(locations[m+1][1])], alpha=1)
                        #     # elif rlRef == rl or rlRef == otherRL:
                        #     #     self.ax.plot([float(locations[m][0]), float(locations[m+1][0])], [float(locations[m][1]), float(locations[m+1][1])], alpha=0.5)
                        #     # elif rRef == route:
                        #     #     self.ax.plot([float(locations[m][0]), float(locations[m+1][0])], [float(locations[m][1]), float(locations[m+1][1])], alpha=0.3)
                        #     else:
                        #         self.ax.plot([float(locations[m][0]), float(locations[m+1][0])], [float(locations[m][1]), float(locations[m+1][1])], alpha=0.1)
                        # plt.show(block=False)
                        # plt.pause(0.01)


                        # If the line segments intersect, return otherRL to signify that it's an intersection RL
                        if self.doIntersect(routeLink.iloc[l], routeLink.iloc[l+1], otherRouteLink.iloc[otherL], otherRouteLink.iloc[otherL+1]):
                            # self.ax.cla()
                            # self.ax.plot([float(routeLink.iloc[l][0]), float(routeLink.iloc[l+1][0])], [float(routeLink.iloc[l][1]), float(routeLink.iloc[l+1][1])], 'v')
                            # self.ax.plot([float(otherRouteLink.iloc[otherL][0]), float(otherRouteLink.iloc[otherL+1][0])], [float(otherRouteLink.iloc[otherL][1]), float(otherRouteLink.iloc[otherL+1][1])], '^')
                            # # self.ax.plot(float(otherRouteLink.iloc[otherL]), float(otherRouteLink[otherL+1]))
                            # self.ax.set_title("Intersection")
                            # plt.show(block=False)
                            # plt.pause(2)
                            return otherRL
                        
            return None
    
    # Given three collinear points p, q, r, the function checks if 
    # point q lies on line segment 'pr' 
    def onSegment(self, p, q, r):
        if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
            return True
        return False
    
    def orientation(self, p, q, r):
        # to find the orientation of an ordered triplet (p,q,r)
        # function returns the following values:
        # 0 : Collinear points
        # 1 : Clockwise points
        # 2 : Counterclockwise
        
        # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
        # for details of below formula. 
        
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if (val > 0):
            
            # Clockwise orientation
            return 1
        elif (val < 0):
            
            # Counterclockwise orientation
            return 2
        else:
            
            # Collinear orientation
            return 0
    
    # The main function that returns true if 
    # the line segment 'p1q1' and 'p2q2' intersect.
    def doIntersect(self,p1,q1,p2,q2):
        
        # Find the 4 orientations required for 
        # the general and special cases
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
    
        # General case
        if ((o1 != o2) and (o3 != o4)):
            return True
    
        # Special Cases
    
        # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
        if ((o1 == 0) and self.onSegment(p1, p2, q1)):
            return True
    
        # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
        if ((o2 == 0) and self.onSegment(p1, q2, q1)):
            return True
    
        # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
        if ((o3 == 0) and self.onSegment(p2, p1, q2)):
            return True
    
        # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
        if ((o4 == 0) and self.onSegment(p2, q1, q2)):
            return True
    
        # If none of the cases
        return False
    
    ####################################################################################################################################




    ############################################### Arrival computing methods #############################################################

    # timeGapDiscontThresh is the number of minutes for which a time gap between two locations exceeding this amount implies the locations are part of separate journeys
    # arrivalRadius dictates a radius for which two intersection RLs being inside of it should be dealt with with caution
    # diffRouteThreshold is a radius for which a bus with no routelinks within this threshold is considered to be on a different route

    # Keys in journeysList are vehicle references, and the value corresponding to each vehicleReference, 'journey', is a dictionary describing the journey of that vehicle
        # Keys in 'journey' include the routes on which the vehicle travelled, and a string 'routeOrder'
        # Each time vehicle ref moves onto a new route (from switching directions or otherwise), the new route's reference is appended to journey['routeOrder']
        # Under each journey['routeRef'] is a list describing the vehicle journey along the route
            # Each index in jounrey['routeRef'] is a tuple, 'arrival', with arrival[0] being the status of the bus, and arrival[1] being the time when the bus was at that status
    # Overall, journeysList -> VehicleRef -> [['routeOrder'->list], ['routeRef'->list of arrivals->[status, time]]]
    def computeArrivals(self, lineDataDir, timeGapDiscontThresh, arrivalRadius, diffRouteThreshold):

        routeDF = pd.read_csv(os.path.join(lineDataDir, 'Routes.csv'), index_col=0)
        routeLinkDF = pd.read_parquet(os.path.join(lineDataDir, 'RouteLinks.parquet'))
        intersectionRLsDF = pd.read_csv(os.path.join(lineDataDir, 'IntersectionRLs.csv'), index_col=0)
        journeyCodeDF = pd.read_csv(os.path.join(lineDataDir, 'JourneyCodes.csv'), index_col=0)
        rawLocationDF = pd.read_parquet(os.path.join(lineDataDir, 'RawBusLocations.parquet'))

        # arrivalDFDict is a dictionary containing the arrival times DF for the route corresponding to the dict key
        arrivalDFDict = dict()
        for r in routeDF.index:

            columns = []
            for rl in routeDF.loc[r, 'RouteLink0':].dropna():
                columns.extend([str((rl,'0')), str((rl,'1'))])
            
            arrivalDFDict[r] = pd.DataFrame(dtype='object', columns=['JourneyCode', 'VehicleRef'] + columns)


        
        # Keys in journeysList are vehicle references, and the value for that key is  the journey for that vehicleRef
        journeysList = dict()

        # Initialize lists to be used in the computing of arrival times
        prevStatus = dict.fromkeys(rawLocationDF.columns, None)

        # rawLocationDF.iloc[0:100].swifter.apply(lambda row: self.evaluateRow(row, prevStatus, journeyCodeDF, routeDF, routeLinkDF, intersectionRLsDF, arrivalRadius, diffRouteThreshold), axis=1)
        # rawLocationDF.iloc[0:100].swifter.apply(lambda row: self.evaluateRow(row, prevStatus, journeyCodeDF, routeDF, routeLinkDF, intersectionRLsDF, arrivalRadius, diffRouteThreshold), axis=1)
        # rawLocationDF.apply(lambda row: self.function(row), axis=1)
        # exit()

        for r in rawLocationDF.index:
        # def function(self, row):
            
            if rawLocationDF.index.get_loc(r) % 1000 == 0:
                print(rawLocationDF.shape[0], rawLocationDF.index.get_loc(r))
            # print(rawLocationDF.shape[0], rawLocationDF.index.get_loc(r))

            # If the time between the time at the current row of data and the previous row is large, then don't continue the journey of the buses
            time = dateutil.parser.parse(r)
            
            try:
                if time - prevTime > datetime.timedelta(minutes=int(timeGapDiscontThresh)):
                    prevStatus = dict.fromkeys(rawLocationDF.columns, None)
                    for j in journeysList:
                        arrivalDFDict = self.recordJourneyInDF(journey=journeysList[j], vehRef=vehRef, arrivalDFDict=arrivalDFDict)
                    journeysList = dict()

            except NameError:
                pass

            # Converts the raw longitudinal and latitudinal values of buses to discrete 'statuses' describing their progress along the route
            # statuses = self.evaluateRow(rawLocationDF.loc[r], prevStatus, journeyCodeDF, routeDF, routeLinkDF, intersectionRLsDF, arrivalRadius, diffRouteThreshold)
            statuses = self.evaluateRow(rawLocationDF.loc[r], prevStatus, journeyCodeDF, routeDF, routeLinkDF, intersectionRLsDF, arrivalRadius, diffRouteThreshold)

            # TO-DO: updateJourneys SHOULD BASICALLY DO WHAT IS BEING DONE BELOW
            # journeysList = self.updateJourneys(statuses, journeysList, arrivalDFDict)


            # Decide how to update each journey based on new statuses
            # Four general cases - bus is either starting its overall journey, bus is as same status as previous status, bus has moved to some other status, or bus has stopped being recorded
            for i in statuses:

                vehRef, journeyCode = i.split(sep=' ')
                    
                routeRef = journeyCodeDF['RouteRef'][int(journeyCode)]

                # If there is no known data for the column i
                if statuses[i] == None:

                    # Bus is not running and doesn't have a journey being recorded
                    # Or, bus vehRef is currently running, but its status is not currently being posted to i, as bus' location is being recorded in another i
                    if vehRef not in journeysList or journeysList[vehRef]['route'] != routeRef:
                        continue

                    # Statuses have stopped being posted and i is a journeyCode corresponding to the route that the bus was last observed travelling along
                    elif journeysList[vehRef]['route'] == routeRef:

                        # If there has been a big enough gap since the bus' last recording, record and end the journey
                        timeGap = time - journeysList[vehRef]['lastReading']
                        if timeGap > datetime.timedelta(minutes=int(timeGapDiscontThresh)):
                            arrivalDFDict = self.recordJourneyInDF(journey=journeysList[vehRef], vehRef=vehRef, arrivalDFDict=arrivalDFDict)
                            journeysList.pop(vehRef)

                # There is a status for column i
                else:

                    if vehRef in journeysList and journeysList[vehRef]['journeyCode'] != journeyCode and journeysList[vehRef]['route'] == routeRef:
                        # print("Journey code changed but route ref didn't, just to lyk buddy ol pal")
                        print(journeysList[vehRef]['journeyCode'], " changed to ", journeyCode)
                    
                    nextLocation = self.getNextLocation(routeDF.loc[routeRef, 'RouteLink0':], statuses[i])

                    # Bus location is known and its journey has not begun being recorded, but it's at last stop
                    # No point in creating a journey from this as there are no future arrival times to record
                    if nextLocation[1] == '2' and vehRef not in journeysList:
                        continue

                    # Bus location is known and its journey has not begun being recorded, and it's not at last stop
                    # Initialize new journey for bus in journeysList
                    elif vehRef not in journeysList:
                        # Initialize dictionary for journey
                        # Don't add first status - bus could be at very end of status -> not a good representation of arrival time
                        # journeysList[vehRef] = {'Journey':[], 'route':routeRef, 'lastReading':time}
                        journeysList[vehRef] = {'Journey':[], 'route':routeRef, 'journeyCode':journeyCode, 'lastReading':time}

                    # Bus has changed route - record journey along old route and initialize journey along new route
                    elif routeRef != journeysList[vehRef]['route']:
                        arrivalDFDict = self.recordJourneyInDF(journey=journeysList[vehRef], vehRef=vehRef, arrivalDFDict=arrivalDFDict)
                        journeysList[vehRef] = {'Journey':[], 'route':routeRef, 'journeyCode':journeyCode, 'lastReading':time}

                    # If journey has not been initialized but isn't being recorded yet (i.e. first arrival for this bus received)
                    elif len(journeysList[vehRef]['Journey']) == 0:
                        # This happens when bus is moving backwards through route
                            # To begin with, program will assign nearest location as bus status
                            # As bus moves backwards through route, program will ignore earlier locations when finding nearest location
                            # This results in the bus getting further way from the potential route links
                            # Eventually, the bus is over diffRouteThreshold distance from the nearest route links, so program assigns None status
                        if prevStatus[i] == None:
                            journeysList.pop(vehRef)
                            
                        # If status has changed since first received status for this vehicle, record this arrival time
                        # Also ensure that bus has not moved backwards in route, just in case
                        elif statuses[i] != prevStatus[i] and statuses[i] != self.getEarlierLocation(statuses[i], prevStatus[i], routeDF.loc[routeRef]):
                            journeysList[vehRef]['Journey'].append([statuses[i], time])

                    # If current status is the same as the most recent status
                    # Don't have to do anything here
                    elif statuses[i] == journeysList[vehRef]['Journey'][-1][0]:
                        # Happens if status has not changed, but routeRef has. i.e. means two distinct routes share the same route link
                        if routeRef != journeysList[vehRef]['route']:
                            print("Bruh!")
                            exit()


                    # We know that statuses[i] != journeysList[vehRef]['Journey'][-1][0] due to above case
                    # So, if statuses[i] is returned from the getEarlierLocation(statuses[i], journeysList[vehRef]['Journey'][-1][0]), it is strictly earlier than the last recorded location
                    # Thus, bus moved backwards in route - should stop the journey in this case
                    elif statuses[i] == self.getEarlierLocation(statuses[i], journeysList[vehRef]['Journey'][-1][0], routeDF.loc[routeRef]) and statuses[i][1] != '-1':
                        # self.plotJourney(journeysList[vehRef], vehRef, routeLinkDF, routeDF, rawLocationDF, journeyCodeDF, journeysList[vehRef][journeysList[vehRef]['routeOrder'][-1]][-1][0][0], statuses[i][0], r)
                        arrivalDFDict = self.recordJourneyInDF(journey=journeysList[vehRef], vehRef=vehRef, arrivalDFDict=arrivalDFDict)
                        journeysList.pop(vehRef)

                    
                    # Status is now a status that is different to and not earlier in the route than the previously recorded status i.e. bus has moved along the route
                    else:

                        # Check if status is next status in the route
                        # If it is, simply append new status to journey
                        if statuses[i] == nextLocation:
                            journeysList[vehRef]['Journey'].append([statuses[i], time])

                        # If nextLocation is a '2' location, bus has reached final stop in the route, so record the journey
                        elif nextLocation[1] == '2':
                            journeysList[vehRef]['Journey'].append([statuses[i], time])
                            arrivalDFDict = self.recordJourneyInDF(journey=journeysList[vehRef], vehRef=vehRef, arrivalDFDict=arrivalDFDict)
                            journeysList.pop(vehRef)

                        # If bus goes from any location other than a '-1' location back to a '-1' location, remove all previous locations
                        # Bus can sometimes move out of '-1' location radius if arrivalRadius is small enough
                        # - We assume bus cannot move backwards through route
                        elif statuses[i][1] == '-1':
                            journeysList[vehRef]['Journey'] = []
                            

                        # If status isn't next immediate status in route, fill in gap between statuses, assuming the gaps between arrival times are evenly spaced
                        else:
                            gapJourney = self.getJourneyBetweenLocations(location1=journeysList[vehRef]['Journey'][-1][0], location2=statuses[i], route=routeDF.loc[routeRef])
                            # Do [1:] since first arrival in the return list is already recorded in the dictionary
                            timings = self.splitTimeEvenly(gapJourney, journeysList[vehRef]['Journey'][-1][1], time)[1:]
                            journeysList[vehRef]['Journey'].extend(timings)


                    # Update the 'lastReading' time every time
                    # Keeps buses from being removed if the bus status is known
                    # Buses are removed from journeysList if a status isn't received for timeGapDiscontThresh time
                    if vehRef in journeysList:
                        journeysList[vehRef]['lastReading'] = time



            prevTime = time

            prevStatus = deepcopy(statuses)


        if os.path.exists(os.path.join(lineDataDir, 'arrivals')) == False:
            os.mkdir(os.path.join(lineDataDir, 'arrivals'))
        for r in arrivalDFDict:
            arrivalDFDict[r].to_parquet(os.path.join(lineDataDir, 'arrivals', r))


    #Returns a list which represents the status of each of the entries in 'locations'
    def evaluateRow(self, locations, prevStatus, journeyCodeDF, routeDF, routeLinkDF, intersectionRLsDF, arrivalRadius, diffRouteThreshold):

        statuses = dict.fromkeys(locations.index, None)

        for n in locations.index:

            journeyCode = int(n[n.index(' ')+1:])
            routeRef = journeyCodeDF['RouteRef'][journeyCode]

            # If there is no location, simply assign None as its status
            if type(locations[n]) == type(None):
                statuses[n] = None

            # The next two if cases deal with the bus being at the last location in its current route

            # A 2 as the second index of the status indicates that the bus has reached the end of the route and hasn't turned off or switched direction yet
            # Continue setting the status with a 2 as the second index until the bus switches direction or turns off
            elif type(prevStatus[n]) != type(None) and prevStatus[n][1] == '2':
                # statuses.append(prevStatus[n])
                statuses[n] = prevStatus[n]

            # This only happens when 'prevStatus[n]' is the last location in the bus' journey along that route
            elif type(prevStatus[n]) != type(None) and self.getNextLocation(routeDF.loc[routeRef], prevStatus[n])[1] == '2':
                # 2 in the second index indicates the bus has come to the end of its journey and is still posting locations
                # The bus' status will continue to be this until it switches direction and stops posting locations for its current direction
                statuses[n] = (prevStatus[n][0], '2')

            # If the bus is not in the last location along its route, figure out where it is using its position, the locations around it, and its previous location
            else:
                location = self.getStatus(locations[n], routeDF.loc[routeRef], prevStatus[n], routeLinkDF, arrivalRadius, diffRouteThreshold, intersectionRLsDF)
                statuses[n] = location

        return statuses


    '''Returns the 'status' of the bus in the form (RL, x), where RL is a route link ref, and x is -1 if the bus is at the 'From' stop of RL and RL is the first route 
    link in the route 0 if the bus is between the 'To' stop of the previous route link in the route and the 'To' stop of RL, and 1 if the bus is at the 'To' stop
    of RL. x can be 2 if the bus has been at the last status in the route (RL, 1) for at least one recording - this is assigned above in the 'evaluateRow' method
    => General bus journey along a route with route links RL1, RL2, ..., RLN is (RL1, -1), (RL1, 0), (RL1, 1), (RL2, 0), (RL2, 1), ..., (RLN, 0), (RLN, 1), (RLN, 2)'''

    ''' Generally, will return the nearest status [(RL, x)] to the bus such that the status is later on in the route than the previous status of the bus.
    However, if at least two 'intersection route links' are detected near the bus, will return the earliest route link in the route.
    An intersection route link RLn is a route link such that another route link RLm in the same route crosses over it. Natually, RLm would also be an intersection
    route link'''
    def getStatus(self, location, route, prevStatus, routeLinks, arrivalRadius, diffRouteThreshold, intersectionRLs):
        
        # The first few columns of info about the route aren't useful here, so slice them out
        route = route['RouteLink0':].dropna()

        # If there is a known previous route link, restrict route DF to only route links from the previous route link onwards
        if prevStatus != None:
            prevStatusRouteLinkCol = route[route==prevStatus[0]].index[0]
            route = route[prevStatusRouteLinkCol:]

        # Data structures to store distances between bus and route links
        allDistances = dict()
        shortestDistances = []
        
        for routeLinkRef in route:

            routeLink = routeLinks.loc[routeLinkRef, 'Location0':].dropna()

            # Route Links are made up of a series of locations describing the structure of the route link
            # Compute the cartesian distances between bus' location, 'location', and each of the locations in routeLink
            distances = [( (l[0]-float(location[0]))**2 + (l[1]-float(location[1]))**2 )**0.5 for l in routeLink]

            # closestDistance is the shortest distance between the bus and a location in routeLink
            # It gives an idea of how close the bus is to the routeLink
            closestDistance = min(distances)
            # Added to allDistances only if the distances is less than some threshold
            # Essentially prevents buses from being mapped onto the wrong route, if buses from other lines with the same line name manage to get into rawLocationDF somehow
            if closestDistance < diffRouteThreshold:
                allDistances[routeLinkRef] = closestDistance

            # If closestDistance is within arrivalRadius, add it to 'shortestDistances'
            # If the bus is near to multiple route links which intersect eachother, these route links are dealt with carefully later to ensure the right one gets assigned to the location
            if closestDistance < arrivalRadius:

                # If the route link is the first route link in the route, and the distance between the bus and the first location in the route is less than arrivalRadius, the bus is at the first stop
                # A special status containing a '-1' in the second index is given in this case
                if ('RouteLink0' in route) and (route[route==routeLinkRef].index[0]=='RouteLink0') and (distances[0] < arrivalRadius):
                    shortestDistances.append((routeLinkRef, '-1'))

                # (routeLink, 1) means the bus is at the 'To' stop of routeLink
                # (routeLink, 0) means the bus is on routelink, but not yet at the stop
                elif distances[len(distances)-1] < arrivalRadius:
                    shortestDistances.append((routeLinkRef, '1'))

                # If the previous status stated that the bus was at the stop, and the current one says it isn't, implying it moved backwards, assume it didn't move backwards
                # If this isn't done, the fixPrevArrivalTimes method messes up, as it tries to fill in the gap by moving forwards through the route - so it will never reach an earlier location from a later one
                elif type(prevStatus) != type(None) and prevStatus[0] == routeLinkRef and prevStatus[1] == '1':
                    shortestDistances.append((routeLinkRef, '1'))
                else:
                    shortestDistances.append((routeLinkRef, '0'))


        # Cases where buses are near route links that intersect eachother are dealt with carefully here

        # If there's one or less intersection route links in arrival radius, continue and use the closest route link in the radius
        # Otherwise, if there's more than one intersection route link detected in the radius, return the earlier one
        # If bus is on later routelink on intersection point, earlier routelink will have been sliced out of potential routelinks anyway, so this won't happen
        # Thus, problems only arise when there is no previous status to use to slice the potential route links. In these cases, wait until a status can be confidently assigned
        nearbyIntersectionRLs = []
        for rl in shortestDistances:

            # IF IT'S ONLY NEAR ONE INTERSECTION RL, SURELY IT'S OK TO ASSIGN THE RL EVEN IF IT'S THE FIRST ASSIGNMENT??????
            #If it's the first assignment of location and bus is at intersection point, just wait until it's off it to assign an initial location
            # if rl[0] in intersectionRLs and type(prevStatus) == type(None):
            #     return None
            
            # Adds rl to 'nearbyIntersectionRLs' to show that there is at least one intersection routelink within arrival radius
            # If another intersection routelink is detected, then measures will be taken
            if rl[0] in intersectionRLs and len(nearbyIntersectionRLs) == 0:
                nearbyIntersectionRLs.append(rl)

            # If an intersection routelink has already been detected within arrival radius and another one is within the radius...
            elif rl[0] in intersectionRLs and len(nearbyIntersectionRLs) != 0:
                nearbyIntersectionRLs.append(rl)

            # If there is more than one intersection routelink within arrivalRadius, return the earliest routelink in 'nearbyIntersectionRLs'
            if len(nearbyIntersectionRLs) >= 2:
                earlierRL = self.getEarlierLocation(route, nearbyIntersectionRLs)
                return earlierRL

        # Return closest route link in shortestDistances
        if len(shortestDistances) != 0:
            return min(shortestDistances, key=lambda x: allDistances[x[0]])
        
        # If there are no route links within diffRouteThreshold of the bus location, it's likely the bus does not travel along the stored routes, so return no status
        if len(allDistances) == 0:
            return None
        
        # If no route links are within arrivalRadius distance of the bus, there will be no route links in shortestDistances
        # If there are routelinks within the diffRouteThreshold and thus in allDistances, return the nearest route link in the routelinks in allDistances
        else:
            closestRL = min(allDistances, key=lambda x: allDistances[x])
            # If the previous status was that the bus was at the same route link as the one nearest to it now, but was at the 'To' stop, assume this is still the case
            if type(prevStatus) != type(None) and prevStatus == (closestRL, '1'):
                return (closestRL, '1')
            # Otherwise, the stop is clearly outside arrivalRadius distance of the bus, as the entire route link is outside arrivalRadius distance of the bus
            else:
                return (closestRL, '0')
            
    # Passes the location that comes after 'location' in 'route'
    # route should be a pandas Series. The series can include direction and null values
    def getNextLocation(self, route, location):
        # Can quickly deal with every case where there isn't a '1' as the second index of location
        if location[1] == '0':
            return (location[0], '1')
        elif location[1] == '2':
            return location
        elif location[1] == '-1':
            return (location[0], '0')
        

        # Program gets to here only if there is a '1' as the second index of location

        # Important that route has no null values for the next step
        route = route.dropna()
        
        # Find the index of the current status in the route
        rlIndex = list(route).index(location[0])
        try:
            # Try to find next route link in the route
            nextRL = list(route)[rlIndex+1]

        # If there is no next route link, will throw IndexError, and thus the current route link is the final route link
        except IndexError:
            return (location[0], '2')

        # If there is a next route link, return it
        return (nextRL, '0')
    
    # time1 should be time for first index in locations, time2 should be time for last index in locations
    # Times for each index between them will be split evenly between time1 and time2
    def splitTimeEvenly(self, locations, time1, time2):
        timeGap = time2 - time1
        timing = []
        time = time1
        for l in locations:
            timing.append([l, time])
            time += timeGap/(len(locations)-1)

        return timing

    # Sometimes, a bus can be in a location L at time t, and then at the next time t+1, have skipped the next location in the route to a further location
    # i.e. 'skip' a location along the route
    # When this happens, this function 'bridges the gap' between the two locations to restore continuity to the arrivals
    def fixDiscontinuities(self, arrivalsDict, statuses, prevStatus, routeDF, index, routeLinkDF):

        # for n in range(len(prevStatus)):
        for n in prevStatus:
            # If there is no previous status, or the bus has switched off since the last recording, or there has been no change to the status, there are no discontinuities
            if prevStatus[n] == None or statuses[n] == None or prevStatus[n] == statuses[n]:
                continue

            route = routeDF.loc[n[n.index(' ')+1:]].dropna()

            # If the location goes from some location other than the starting point back to the starting point, set all previous locations since the bus turned on to the start point
            if statuses[n][1] == '-1' and prevStatus[n][1] != '-1':
                m = len(arrivalsDict[n])-1
                # while arrivalsDict[m][n] != None and m >= 0:
                while arrivalsDict[n][m] != None and m >= 0:
                    # arrivalsDict[m][n] = statuses[n]
                    arrivalsDict[n][m] = statuses[n]
                    m -= 1
                continue

            # i.e. if the location of the bus has changed to a location other than the next one in the route
            # Cases where bus moves backwards are dealt with in 'getStatus' by restricting potential routelinks to only later ones, so don't need to worry about it here
            # Only case to deal with here is when the bus skips locations along the route
            # When this happens, put all locations between its current location and the previous one as the previous status, including the previous status
            if prevStatus[n] != statuses[n] and statuses[n] != self.getNextLocation(route, prevStatus[n]):

                skipped = self.getJourneyBetweenLocations(prevStatus[n], statuses[n], route)

                # Plot situation
                # if len(skipped) >= 3:
                #     print("Gap in one time step is long: ", prevStatus[n], statuses[n])

                    # prevIndex = self.rawLocationDF.index.get_loc(index)-1
                    # prevLocations = self.rawLocationDF.iloc[prevIndex].dropna()
                    # curLocations = self.rawLocationDF.loc[index].dropna()
                    # routeLinkBounds = (prevStatus[n][0], statuses[n][0])
                    # fig = plt.gcf()
                    # ax = fig.gca()
                    # try:
                    #     while True:
                    #         for locations in [prevLocations, curLocations]:
                    #             ax.clear()
                    #             ax.plot([l[0] for l in locations], [l[1] for l in locations], 'o', label=locations.name)
                    #             for rl in routeLinkDF.loc[routeLinkBounds[0]:routeLinkBounds[1]].index:
                    #                 routeLink = routeLinkDF.loc[rl]['Location 0':].dropna()
                    #                 # print(routeLink)
                    #                 ax.plot([l[0] for l in routeLink], [l[1] for l in routeLink], alpha=0.5, label=rl)
                    #             fig.legend()
                    #             plt.show(block=False)
                    #             plt.pause(0.5)
                    # except KeyboardInterrupt:
                    #     pass

                # arrivalsDict[len(arrivalsDict)-1][n] = skipped
                arrivalsDict[n][len(arrivalsDict[n])-1] = skipped

        # print(arrivalsDict)
        # exit()

        return arrivalsDict
    
    # Returns sequence of locations between location1 and location2, including location1 and location2
    # location1 must be an earlier location than location2, otherwise the method will not work
    def getJourneyBetweenLocations(self, location1, location2, route):
        if location1 == location2:
            return None
        
        location = location1
        journey = [location]

        while True:
            # Occurs if location is last location in route
            if location == self.getNextLocation(route, location):
                # print("Reached end of journey before next stop", location1, location2)
                return journey
            
            location = self.getNextLocation(route, location)
            journey.append(location)

            if location == location2:
                if len(journey) == 1:

                    print(journey, location1, location2)
                return journey

            # if location != location2:
            #     journey.append(location)
            # # Once location reaches location2, return the journey
            # else:
            #     return journey
        
    # 
    def recordJourneyInDF(self, journey, vehRef, arrivalDFDict):
        # If journey is just arrival at one stop, this is not useful, so don't record it
        if len(journey['Journey']) <= 1:
            return arrivalDFDict
        
        else:
            # Get DF corresponding to the journey's route, and initialize a new row in it for the journey
            routeRef = journey['route']
            journeyDF = arrivalDFDict[routeRef]
            index = 'Journey '+str(journeyDF.shape[0]+1)
            journeyDF.loc[index] = pd.Series(data=[journey['journeyCode'], vehRef] + [None]*(journeyDF.shape[1]-2), index=journeyDF.columns, dtype='object')

            for s in journey['Journey']:
                if s[0][1] != '-1' and s[0][1] != '2':
                    journeyDF[str(s[0])][index] = s[1]

            arrivalDFDict[routeRef] = journeyDF
            return arrivalDFDict
    
    def plotJourney(self, journey, vehRef, routeLinkDF, routeDF, rawLocationsDF, journeyCodeDF, lastRL, backRL, time):
        print("Plotting journey")
        fig = plt.gcf()
        ax = fig.gca()
        ax.clear()
        for r in journey['routeOrder']:
            for s in journey[r]:

                ax.clear()

                try:
                # print("Looking for column")
                # Plot bus location
                # location = rawLocationsDF[vehRef+' '+r][str(s[1].date())+'T'+str(s[1].time())+'+00:00']
                    for l in rawLocationsDF.loc[str(s[1].date())+'T'+str(s[1].time())+'+00:00'].dropna().index:
                        if vehRef in l:
                            # print("Column found!")
                            index = rawLocationsDF.index.get_loc(str(s[1].date())+'T'+str(s[1].time())+'+00:00')
                            # print("I'm confused cos there is index", index)
                            # print(index)
                            location = rawLocationsDF[l][str(s[1].date())+'T'+str(s[1].time())+'+00:00']
                            ax.plot(float(location[0]), float(location[1]), 'v', label=l)

                    route = routeDF.loc[r, 'RouteLink0':].dropna()
                    # rlRef = s[0][0]

                    # Plot every stop other than relevant routelink with a low alpha
                    for rl in route:
                        # if rl == rlRef:
                        if rl == lastRL or rl == backRL:
                            continue
                        ax.plot([l[0] for l in routeLinkDF.loc[rl, 'Location0':].dropna()], [l[1] for l in routeLinkDF.loc[rl, 'Location0':].dropna()], alpha=0.3)

                    # Plot relevant routelink with high alpha
                    # ax.plot([float(l[0]) for l in routeLinkDF.loc[rlRef, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[rlRef, 'Location0':].dropna()], label=rlRef)

                    ax.plot([float(l[0]) for l in routeLinkDF.loc[lastRL, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[lastRL, 'Location0':].dropna()], 'o')
                    ax.plot([float(l[0]) for l in routeLinkDF.loc[backRL, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[backRL, 'Location0':].dropna()], 'v')

                    fig.legend()
                    ax.set_title(str(s[1].time().strftime(format='%H:%M:%S')))
                    plt.show(block=False)
                    plt.pause(0.3)

                except KeyError:
                    pass
        
        try:
            # for i in range(index, index+15):
            while rawLocationsDF.index[index] != time:
                ax.clear()
                # print("There is index")
                for l in rawLocationsDF.iloc[index].dropna().index:
                    if vehRef in l:
                        journeyCode = int(l[l.index(' ')+1:])
                        routeRef = journeyCodeDF['RouteRef'][journeyCode]
                        location = rawLocationsDF.iloc[index][l]
                        ax.plot(float(location[0]), float(location[1]), 'v', label=l)

                route = routeDF.loc[routeRef, 'RouteLink0':].dropna()
                # rlRef = s[0][0]

                # Plot every stop other than relevant routelink with a low alpha
                for rl in route:
                    if rl == backRL or rl == lastRL:
                        continue
                    ax.plot([float(l[0]) for l in routeLinkDF.loc[rl, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[rl, 'Location0':].dropna()], alpha=0.3)

                ax.plot([float(l[0]) for l in routeLinkDF.loc[lastRL, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[lastRL, 'Location0':].dropna()], 'o', zorder=-10)
                ax.plot([float(l[0]) for l in routeLinkDF.loc[backRL, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[backRL, 'Location0':].dropna()], '^', zorder=-10)

                # Plot relevant routelink with high alpha
                # ax.plot([float(l[0]) for l in routeLinkDF.loc[rlRef, 'Location0':].dropna()], [float(l[1]) for l in routeLinkDF.loc[rlRef, 'Location0':].dropna()], label=rlRef)

                fig.legend()
                ax.set_title(rawLocationsDF.index[index])
                plt.show(block=False)
                plt.pause(0.5)

                index += 1

        except UnboundLocalError:
            # print("No index to use")
            pass

    # route should be Series of route, not DataFrame
    def getEarlierLocation(self, location1, location2, route):
        # print("Which is earlier?:",location1, location2)
        if location1[1] == '-1':
            return location1
        elif location2[1] == '-1':
            return location2
        
        if location1[0] == location2[0]:
            if location1[1] == '1':
                # print("Earlier:",location2)
                return location2
            else:
                # print("Earlier:",location1)
                return location1
        
        index1 = list(route).index(location1[0])
        index2 = list(route).index(location2[0])

        if index1 > index2:
            # print("Earlier:",location2)
            return location2
        else:
            # print("Earlier:",location1)
            return location1

    ###################################################################################################################################

# atg = ArrivalTimesGetter()
# atg.computeArrivals(lineDataDir='F:\CS344Stuff\App_Versions\!!!FINAL\.Application\BusModeller\data\linedata\LineX18', timeGapDiscontThresh=2, arrivalRadius=0.0003, diffRouteThreshold=0.01)
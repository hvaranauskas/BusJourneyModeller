import numpy as np
import pandas as pd
import scipy
import random as rand
import pickle
import matplotlib.pyplot as plt
import datetime
import os
import dateutil
from copy import deepcopy


class RandomWalkModeller():

    def plotArrivalTimes(self, arrivalTimes):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for j in arrivalTimes.index:
            # pd.Timestamp.to
            ax.plot([str(c) for c in arrivalTimes.loc[j].dropna().index], [datetime.datetime.combine(datetime.date.min, i.to_pydatetime().time()) for i in arrivalTimes.loc[j].dropna()])
        plt.show()

    #Returns transition matrix which attempts to model journeys of the buses along the route with the given departureTime and timeStepInterval
    def modelRoute(self, arrivalsDataDir, departureTime, timeStepInterval, groupJourneys=True, threshold=None, routeDirection=None, routeDir=None, stopsOnly=False, store=False, skipFree=True, useStoredVectors=False, plotProbVector=False):

        self.dir = './cache/'+departureTime.strftime('%H%M%S')+'_'+str(timeStepInterval.total_seconds())+'_'+str(groupJourneys)+'_'+str(stopsOnly)+'_'+routeDirection+'/'

        arrivalTimes = pickle.load(open(arrivalsDataDir, 'rb'))

        # IF YOU'RE NOT ASSUMING MARKOV, THEN HAVE TO ASSUME THAT JOURNEYS WITH SIMILAR DEPARTURE TIMES HAVE SIMILAR TRANSITION TIME DISTS
        # Journeys are similar if they're within threshold 'threshold'
        self.ArrivalsDF = self.formatArrivalsDF(arrivalTimes, departureTime, timeStepInterval, stopsOnly, routeDir, routeDirection, groupJourneys, threshold)

        pvc = ProbabilityVectorCalculator()
        probVectors = pvc.computeProbVectors(self.ArrivalsDF, departureTime, timeStepInterval, stopsOnly, plotProbVector)
            
        if store == True:
            self.dir = './cache/'+departureTime.strftime('%H%M%S')+'_'+str(timeStepInterval.total_seconds())+'_'+str(groupJourneys)+'_'+str(stopsOnly)+'_'+routeDirection+'/'
            if os.path.exists(self.dir) == False:
                if os.path.exists('./cache')==False:
                    os.mkdir('./cache')
                os.mkdir(self.dir)

            pickle.dump(probVectors, open(self.dir+'probVectors', 'wb'))

        transitionMatrix, params = self.fitTransitionMatrix(probVectors, skipFree, store)

        self.plotWalk(params=params, probabilityVectors=probVectors)

        return params, probVectors
    

    # Does a number of useful things
    # Can round times in DF to nearest time interval 'timeStepInterval' relative to 'departureTime'
        # e.g. if departure time is 18:57, and interval is 5 mins, then times will be rounded to 18:47, 18:52, 18:57, 19:02 etc
    # Can get departure times that are within some time 'threshold' of 'departureTime'
        # e.g. if given departure time is 17:03, and threshold is 10 minutes, journey departure times between 16:53 and 17:13 will be returned
    # Can restrict DF to only stop locations (only (RL, 1) locations), as opposed to route link and stop locations ((RL, 0) and (RL, 1) locations)
    # Can cut off start and last location from route - increases number of 'complete' journeys, as sometimes the only arrivals missing are the start or stop journey
    def formatArrivalsDF(self, 
                         arrivalTimes, 
                         departureTime, 
                         timeStepInterval=None, 
                         stopsOnly=False, 
                         routeDir=None, 
                         routeDirection=None, 
                         groupJourneys=False, 
                         zScoreThreshold=None,
                         snipEdgeLocations=False
                         ):

        if type(departureTime) == datetime.datetime:
            departureTime = departureTime.time()
        elif type(departureTime) == datetime.time:
            pass
        else:
            print("check type of departureTime - formatArrivalsDF")
            print("departureTime has type ", type(departureTime))
            exit()


        # Slice off 'VehicleRef' column if it exists
        if 'VehicleRef' in arrivalTimes:
            arrivalTimes = arrivalTimes.drop('VehicleRef', axis=1)


        # Convert dtype to datetime
        arrivalTimes = arrivalTimes.astype('datetime64')


        if routeDir == None and routeDirection != None:
            print("You need to give route dir with direction")
            exit()


        # Restricts arrivalTimes to only stops which are on the desired route
        elif routeDir != None:
            routeDF = pd.read_csv(routeDir, index_col=0)
            routeDF = routeDF[routeDF['Direction']==routeDirection].iloc[0]['RouteLink0':].dropna()
            arrivalTimes = arrivalTimes[[c for c in arrivalTimes.columns if c[0] in routeDF.values]]


        # If computing for stops only, restrict DF to only stops
        if stopsOnly == True:
            columns = [arrivalTimes.columns[0]] + [c for c in arrivalTimes.columns if c[1] != '0']
            arrivalTimes = arrivalTimes[columns]


        # SHOULD THIS BE DONE?
        # NO - RESTRICTS DF TO ONLY JOURNEYS WHICH LEAVE AT SIMILAR TIME
        # ALTERNATIVE WOULD BE TO RESTRICT (STOP -> ROUTE LINK) TRANSITIONS TO LOW END, WHICH IS WHY THIS WAS DONE

        # CHANGED SO IT DOESNT DO THIS
        # JOURNEYS WILL JOIN ON LATER IF THEY'RE RELEVANT, ANYWAY
        # if groupJourneys == True:

        #     firstStop = arrivalTimes.columns[0]

        #     # Compute which journeys (if any) have a departure time within the threshold specified above of the actual departure time
        #     temp = pd.DataFrame(columns = arrivalTimes.columns)
        #     firstCol = arrivalTimes[firstStop].dropna()
        #     for j in firstCol.index:
        #         journeyDepTime = datetime.datetime.combine(datetime.date.min, arrivalTimes[firstStop][j].time())
        #         difference = datetime.datetime.combine(datetime.date.min, departureTime) - journeyDepTime

        #         if abs(difference) <= threshold:
        #             temp.loc[j] = arrivalTimes.loc[j]
        #             # temp.loc[j] = arrivalTimes.loc[j] + difference

        #     arrivalTimes = temp

        # If there is no interval to round the arrival times to, just return the DF
        if timeStepInterval == None:
            return arrivalTimes


        # Do this to know how much to subtract from arrival times DF so the values to intervals relative to the departure time
        # e.g. if we want for the bus to depart at 10:10:09 and for the time step to be 30 seconds, want to round times to 10:10:09, 10:10:39, 10:11:09, 10:11:39 etc
        # roundedDepartureTime = self.roundTimeToNearestInterval(departureTime, timeStepInterval)
        # temp1 = datetime.datetime.combine(datetime.date.min, departureTime)
        # temp2 = datetime.datetime.combine(datetime.date.min, roundedDepartureTime)
        # delta = temp1 - temp2

        # # Round datetimes in arrivalTimes DF to the nearest defined interval
        # arrivalTimes = arrivalTimes - delta
        # for c in arrivalTimes:
        #     arrivalTimes[c] = (arrivalTimes[c].dt.round(timeStepInterval) + delta).dt.time

        # Compute z score thresholds, if they are given
        if zScoreThreshold != None:
            waitTimeThresholds = dict()
            for c in range(arrivalTimes.shape[1]-1):
                if arrivalTimes.columns[c][1] == '1':
                    temp = arrivalTimes.iloc[:, c][(pd.isna(arrivalTimes.iloc[:, c+1])==False) & (pd.isna(arrivalTimes.iloc[:, c])==False)].index
                    # print(arrivalTimes.loc[temp].iloc[c])
                    # print(arrivalTimes.loc[temp].iloc[c+1])

                    transitTimes = arrivalTimes.loc[temp].iloc[:, c+1] - arrivalTimes.loc[temp].iloc[:, c]
                    mean = transitTimes.mean()
                    std = transitTimes.std()

                    waitTimeThresholds[arrivalTimes.columns[c]] = zScoreThreshold * std + mean

        else:
            waitTimeThresholds = None


        # Rounds time UP to nearest interval relative to departure time - ceilings it
        delta = datetime.datetime.combine(datetime.date(year=2000,month=1,day=1), departureTime) - pd.Timestamp(datetime.datetime.combine(datetime.date(year=2000,month=1,day=1), departureTime)).floor(timeStepInterval)

        for c in arrivalTimes:
            arrivalTimes[c] = ((arrivalTimes[c] - delta).dt.ceil(timeStepInterval) + delta).dt.time

        if waitTimeThresholds == None:
            return arrivalTimes
        
        else:
            return arrivalTimes, waitTimeThresholds
        

    def plotNumJourneysByRange(self, max, departureTime, arrivalTimes):

        tallyDict = dict()
        for n in range(max):
            threshold = datetime.timedelta(minutes=n)
            tallyDict[n] = self.formatArrivalsDF(arrivalTimes, departureTime, threshold=threshold).shape[0]

        fig = plt.gcf()
        ax = fig.gca()

        ax.bar([n for n in tallyDict], tallyDict.values())
        plt.show(block=False)

    def roundTimeToNearestInterval(self, time, interval):

        if type(time) == datetime.time:
            formattedTime = datetime.datetime.combine( datetime.date.min, time )
        elif type(time) == datetime.datetime:
            formattedTime = datetime.datetime.combine( datetime.date.min, time.time() )
            date = time.date()
        else:
            print(time, " is not datetime.datetime or datetime.time")
            exit()

        temp = formattedTime

        while temp > (datetime.datetime.min + interval):
            temp -= interval

        if temp > (datetime.datetime.min + interval/2):
            roundedTime = (formattedTime - temp) + interval
        else:
            roundedTime = (formattedTime-temp)

        if type(time) == datetime.datetime:
            return datetime.datetime.combine(date, datetime.time.min) + roundedTime
        else:
            return (datetime.datetime.min + roundedTime).time()

    def fitTransitionMatrix(self, probabilityVectors):

        self.stopNelderMead = False
        self.loop = 0

        params=[rand.random() for n in range(len(probabilityVectors[0][1])-1)]

        departureTime = probabilityVectors[0][0]
        stepSize = datetime.datetime.combine(datetime.date.min, probabilityVectors[1][0]) - datetime.datetime.combine(datetime.date.min, probabilityVectors[0][0])

        try:
            result = scipy.optimize.minimize(fun=self.lossFunctionLongTerm, # fun=self.lossFunctionMomentToMoment, 
                                                x0=params,
                                                method='Nelder-Mead',
                                                bounds=[(0,1) for x in range(len(params))], 
                                                args=(probabilityVectors),
                                                callback=self.updateProgress
                                                )
        except Exception as e:
            print("EXCEPTION!!", e)
            self.stopNelderMead = False
            return (self.currentParams, self.currentLoss)
        

        return (result.x, result.fun)
    


    # Computes arrival times of buses on route
    def computeArrivalTimes(self, arrivalTimes, departureTime, zScoreThreshold):

        # MAY HAVE TO CHANGE
        # WANT TO GET ROUNDED DEPARTURE TIMES
        # journeysIndex = arrivalTimes[arrivalTimes[arrivalTimes.columns[0]]==departureTime].index
        # temp = rwm.formatArrivalsDF(arrivalTimes, 
        #                             datetime.time(hour=16, minute=50),
        #                             stopsOnly=False,
        #                             routeDirection='outbound',
        #                             routeDir='D:\CS344Stuff\DataExtraction\Data\StopData\ExtractedData/Routes.csv',
        #                             groupJourneys=True,
        #                             threshold=datetime.timedelta(minutes=5)
        #                             )

        # arrivalTimes = arrivalTimes.drop('VehicleRef', axis=1).astype('datetime64')

        trackedJourneys = arrivalTimes[pd.isna(arrivalTimes[arrivalTimes.columns[1]])==False]

        # arrivalTimes = arrivalTimes.drop('VehicleRef', axis=1).astype('datetime64')

        # arrivalTimesDists = dict()
        arrivalTimesDists = pd.DataFrame()
        # Initialize dictionary with departure times from first stop
        # arrivalTimesDists[arrivalTimes.columns[0]] = trackedJourneys[trackedJourneys.columns[0]].values
        arrivalTimesDists[arrivalTimes.columns[0]] = trackedJourneys.iloc[:, 0]

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for n in arrivalTimesDists:
        #     print(n)
        #     print(arrivalTimesDists[n])
        # exit()
        # ax.plot([arrivalTimes.columns[0]]*len(arrivalTimesDists[arrivalTimes.columns[0]]), [n for n in range(len(arrivalTimesDists[arrivalTimes.columns[0]]))], 'o')
        # plt.show()
        # exit()
        # print(trackedJourneys[trackedJourneys.columns[0]].values)
        # exit()
        # print(arrival)

        # Compute transition times between each location
        for c in range(trackedJourneys.shape[1]-2):
            
            # print(trackedJourneys.iloc[:, 2*c])
            # print(trackedJourneys.iloc[:, 2*c+1])

            # If transition from stop to route link, do Z scoring and cut outlyers, then simulate arrival times and compute new set of journeys to consider
            if trackedJourneys.columns[c][1] == '0' and trackedJourneys.columns[c+1][1] == '1':
                transitTimesWithOutliers = trackedJourneys.iloc[:, c+1] - trackedJourneys.iloc[:, c]
                # mean = transitTimesWithOutliers.mean()
                # std = transitTimesWithOutliers.std()
                # Removes journeys for which their stop to route link transition time is an outlyer (their Z score is outside threshold)
                # trackedJourneys = trackedJourneys[abs((transitTimesWithOutliers - mean)/std) < zScoreThreshold]
                # lowestTransitTime = min(transitTimesWithOutliers)
                # trackedJourneys = trackedJourneys[transitTimesWithOutliers <= lowestTransitTime + 2*std ]

                allTransitionTimes = arrivalTimes.iloc[:, c+1] - arrivalTimes.iloc[:, c]
                mean = allTransitionTimes.mean()
                std = allTransitionTimes.std()
                trackedJourneys = trackedJourneys[transitTimesWithOutliers <= mean+std]

            # If it's a transition from a route link onto a stop, then just take arrival times for what they are
            # elif trackedJourneys.columns[2*c+1][1] == '1' and trackedJourneys.columns[2*c+2][1] == '0':
            # print(trackedJourneys.iloc[:, c+1].values)
            # exit()
            # arrivalTimesDists[trackedJourneys.columns[c+1]] = list(trackedJourneys.iloc[:, c+1].values)
            arrivalTimesDists[trackedJourneys.columns[c+1]] = trackedJourneys.iloc[:, c+1]

            # Find other journeys in DF which have arrival times in the range of arrival times in trackedJourneys
            # Do this for both RL to stop transitions, and stop to RL transitions

            maxArrival = max(trackedJourneys.iloc[:, c+1].dt.time)
            minArrival = min(trackedJourneys.iloc[:, c+1].dt.time)

            trackedJourneys = arrivalTimes[(arrivalTimes.iloc[:, c+1].dt.time <= maxArrival) & (arrivalTimes.iloc[:, c+1].dt.time >= minArrival)]
            trackedJourneys = trackedJourneys[pd.isna(trackedJourneys.iloc[:, c+2])==False]

        return arrivalTimesDists



    def updateProgress(self, params=None):
        self.currentParams = params
        self.loop += 1

    def getProgress(self):
        return self.loop, self.currentLoss

    def stopComputing(self):
        self.stopNelderMead = True

    # Returns the times at which the probability of being at a particular location in the probabilityVectors crosses the given probabilityThreshold
    def computeThreshCrossTimes(self, probabilityVectors, probabilityThreshold):

        thresholdTimes = dict()
        for vector in probabilityVectors:

            for l in vector[1]:

                if l in thresholdTimes:
                    continue

                locationIndex = list(vector[1].keys()).index(l)
                prob = sum(list(vector[1].values())[locationIndex:])

                if prob >= probabilityThreshold:
                    thresholdTimes[l] = vector[0]

        return thresholdTimes

    
    def storeTransitionMatrix(self, transitionMatrix):
        with open('transitionMatrix', 'wb') as p:
            pickle.dump(transitionMatrix, p)

    def getStoredTransitionMatrix(self):
        with open('transitionMatrix', 'rb') as p:
            # return self.getTransitionMatrix(pickle.load(p))
            return pickle.load(p)

#================================================ LOSS FUNCTIONS ============================================================
    def lossFunctionAll(self, params, probabilityVectors):

        # Report progress of minimization function
        if self.counter % 500 == 0:
            print("Loop ", self.counter)
        self.counter += 1

        transitionMatrix = self.getTransitionMatrix(params=params)
        totalLoss = 0

        # Initialize initial vector which will be used as a start vector to predict subsequent vectors
        predictions = [list(probabilityVectors[0][1].values())]

        numLosses = 0

        try:
            for n in range(1, len(probabilityVectors)-1):
                
                #Flush probability out of the last index
                # longTermPrediction[len(longTermPrediction)-1] = 0

                testProbVectors[n] = list(probabilityVectors[n][1].values())

                for v in range(len(predictions)):
                    predictions[v] = np.matmul(predictions[v], transitionMatrix)

                predictions.append(actualVector)

                # Compute the error of the prediciton
                # Don't include the first index in this computation
                for prediction in predictions:
                    numLosses += 1
                    totalLoss += sum(abs(prediction[i]-actualVector[i]) for i in range(len(prediction)))

        except IndexError:
            print("Index errr")
            exit()
            
        averageLoss = totalLoss / numLosses
        return averageLoss

    def lossFunctionBoth(self, params, probabilityVectors):

        # Report progress of minimization function
        if self.counter % 500 == 0:
            print("Loop ", self.counter)
        self.counter += 1

        transitionMatrix = self.getTransitionMatrix(params=params)
        totalLoss = 0

        # Initialize initial vector which will be used as a start vector to predict subsequent vectors
        longTermPrediction = list(probabilityVectors[0][1].values())

        # departures =  self.ArrivalsDF[self.ArrivalsDF.columns[0]].value_counts()
        # totalDepartures = sum(departures)

        actualVector = list(probabilityVectors[0][1].values())

        numLosses = 0

        try:
            for n in range(1, len(probabilityVectors)-1):
                
                #Flush probability out of the last index
                # longTermPrediction[len(longTermPrediction)-1] = 0


                shortTermPrediction = np.matmul(actualVector, transitionMatrix)

                # Predict the next vector, using the previous longTermPrediction as a start vector
                longTermPrediction = np.matmul(longTermPrediction, transitionMatrix)

                actualVector = list(probabilityVectors[n][1].values())

                # Compute the error of the prediciton
                # Don't include the first index in this computation
                for prediction in [shortTermPrediction, longTermPrediction]:
                    numLosses += 1
                    totalLoss += sum(abs(prediction[i]-actualVector[i]) for i in range(len(prediction)))

        except IndexError:
            print("Index errr")
            exit()
            
        averageLoss = totalLoss / numLosses
        return averageLoss


    # Initializes an initial vector at the start and doesn't change it, other than introducing probability at the end of every step and removing it at the start
    def lossFunctionLongTerm(self, params, probabilityVectors):

        # Used to stop fitting from UI
        if self.stopNelderMead == True:
            print("Stopping method")
            raise ValueError("Stopped method")

        transitionMatrix = self.getTransitionMatrix(params=params)
        totalLoss = 0

        # Initialize initial vector which will be used as a start vector to predict subsequent vectors
        prediction = list(probabilityVectors[0][1].values())

        # departures =  self.ArrivalsDF[self.ArrivalsDF.columns[0]].value_counts()
        # totalDepartures = sum(departures)

        numLosses = 0

        for n in range(1, len(probabilityVectors)-1):
            
            actualVector = list(probabilityVectors[n][1].values())

            # Predict the next vector, using the previous prediction as a start vector
            prediction = np.matmul(prediction, transitionMatrix)

            # # Add probability to the first index of the vector equal to the ratio of journeys that start at time probabilityVectors[n][0]
            # # Idea behind this coming after prediction is: in a time step, buses should move about the probability already in the vector, then prob should be added after
            # try:
            #     prediction[0] += departures[probabilityVectors[n][0]] / totalDepartures
            # except KeyError:
            #     pass
            
            if abs(sum(prediction) - sum(actualVector)) < 10E-10000:
                print(abs(sum(prediction) - sum(actualVector)))
                print("Don't match up")
                exit()

            # Compute the error of the prediciton
            # Don't include the first index in this computation
            totalLoss += sum(abs(prediction[i]-actualVector[i]) for i in range(len(prediction)))
            numLosses += len(prediction)
        
        averageLoss = totalLoss/numLosses
        self.currentLoss = averageLoss
        return averageLoss

    # Reduces the loss between vectors in probabilityVectors and the vector immediately after it
    def lossFunctionMomentToMoment(self, params, probabilityVectors, skipFree):

        if self.counter % 500 == 0:
            print("Loop ", self.counter)
        self.counter += 1

        transitionMatrix = self.getTransitionMatrix(params=params, skipFree=skipFree)
        totalLoss = 0
        numLosses = 0

        try:
            for n in range(len(probabilityVectors)):
            
                startVector = list(probabilityVectors[n][1].values())

                prediction = np.matmul(startVector, transitionMatrix)

                actualVector = np.array(list(probabilityVectors[n+1][1].values()))

                # In our model, probability is only introduced to the system at the start of the walk
                # Our model only 'takes probability away' from the first index at each time step, as it is biased. Whereas in the data, new probability can be introduced
                # Thus, if we include the first index in error computations, it will make the model less accurate, as it will try to predict the probability introduced to the system
                # Buses departing for their route at time t-1 don't have an effect on buses departing for their route at time t

                # Compute the error of the prediction
                # Don't include the first index when computing the error, for the reasons described above
                totalLoss += sum(abs(prediction[i]-actualVector[i]) for i in range(len(actualVector)))
                numLosses += 1

                # Each probability vector gets multiplied from start to finish by the model
                # This improves the long term performance of the model by ensuring the parameters are fine-tuned
                # THIS MAY OVERFIT THE MODEL, THOUGH
                # predictionVectors.append(list(vector[1].values()))

                # predictionVector = np.matmul(predictionVector, transitionMatrix)
                # actualVector = list(probabilityVectors[probabilityVectors.index(vector)+1][1].values())

                # # Theoretically, each actualVector introduces some new probability to the first location of the route - give this to predictionVector, too
                # predictionVector[0] = actualVector[0]

                # totalLoss += sum(abs(predictionVector[i]-actualVector[i]) for i in range(len(actualVector)))
                
                # For each vector, predict what the next vector will be using the model, then evaluate the error of this prediction
                # for v in range(len(predictionVectors)-1):

                #     prediction = np.matmul(predictionVectors[v], transitionMatrix)
                #     predictionVectors[v] = prediction

                #     actualVector = list(probabilityVectors[probabilityVectors.index(vector)+1][1].values())
                #     totalLoss += sum(abs(prediction[i]-actualVector[i]) for i in range(len(actualVector)))

        except IndexError:
            pass
        
        averageLoss = totalLoss/numLosses
        return averageLoss

    def negative_log_likelihood(self, params, probability_vectors, skipFree):

        transitionMatrix = self.getTransitionMatrix(params=params, skipFree=skipFree)

        nll = 0
        num_vectors = len(probability_vectors)
        
        for vector in probability_vectors:
            vector = list(vector[1].values())
            curr_nll = 0
            print(vector)
            exit()
            
            for i in range(len(vector)-1):
                curr_nll -= np.log(transitionMatrix[vector[i], vector[i+1]])
            
            nll += curr_nll
        
        return nll / num_vectors

    # def goodLossFunction(self, params, thresholdTimes, stepSize, probThreshold=0.95):
    def goodLossFunction(self, params, arrivalTimes, departureTime, stepSize, probThreshold=0.95):

        if self.stopNelderMead == True:
            print("Stopping method")
            raise ValueError("Stopped method")

        transitionMatrix = self.getTransitionMatrix(params)

        # Compute the times at which the probability in the random walk generated by the transitionMatrix pass the probability threshold for each location
        modelThresholdTimes = dict()

        if type(departureTime) == datetime.time:
            time = datetime.datetime.combine(datetime.date.min, departureTime)
        elif type(departureTime) != datetime.datetime:
            print("departureTime should be a datetime or time object!")
            exit()

        vector = np.zeros(len(transitionMatrix))
        vector[0] = 1

        totalLoss = 0

        while len(modelThresholdTimes) < arrivalTimes.shape[1]:

            for l in arrivalTimes.columns:
                if l in modelThresholdTimes:
                    continue

                locationIndex = arrivalTimes.columns.get_loc(l)
                prob = sum(vector[locationIndex:])
                # print(prob)

                if prob >= probThreshold:
                    modelThresholdTimes[l] = time
                    # Compute total loss of the model's prediction of probable arrival times
                    totalLoss += sum([abs((time-datetime.datetime.combine(datetime.date.min, t.time())).total_seconds()) for t in arrivalTimes[l].dropna()])

            vector = np.matmul(vector, transitionMatrix)
            time += stepSize

        averageLoss = totalLoss/sum([arrivalTimes[c].dropna().shape[0] for c in arrivalTimes])
        print(averageLoss)
        return averageLoss


    def getTransitionMatrix(self, params):

        # if skipFree == False:
        #     paramsCopy = list(params).copy()
        #     shape = ((-1+(1+8*len(paramsCopy))**0.5)/2, (-1-(1+8*len(paramsCopy))**0.5)/2)
        #     size = int(max(shape))
        #     transMatrix = []
        #     for n in reversed(range(size)):
        #         temp = [0]*(size-n-1) + paramsCopy[0:(n+1)]
        #         del paramsCopy[0:(n+1)]
        #         transMatrix.append(temp)

        #     return transMatrix

        # Fills diagonal of transition matrix with param values            
        transitionMatrix = np.diag(v=params, k=1)
        # print(transitionMatrix)
        # print(transitionMatrix.shape, len(params))
        # exit()
        np.fill_diagonal(a=transitionMatrix, val=[1-p for p in params], wrap=True)
        transitionMatrix[transitionMatrix.shape[1]-1][transitionMatrix.shape[1]-1] = 1
        return transitionMatrix
    
#==================================== PLOTTING FUNCTIONS ===========================================================
    def startPlotting(self, params, probabilityVectors, showLongTermPrediction, showShortTermPrediction):
        # print("Setting plot var as true")
        self.plotting = True
        self.plotWalk(params, probabilityVectors, showLongTermPrediction, showShortTermPrediction)

    # probabilityVectors can be a list of vectors, or only one
    # If only one vector is given, it will just plot the walk produced by giving this starting vector to the transitionMatrix
    # If all probabilityVectors are given, both the prediction vectors and actual vectors will be plotted. Also, probability will be flushed and introduced to the walk as in the loss function
    def plotWalk(self, params, probabilityVectors=None, showLongTermPrediction=True, showShortTermPrediction=False):

        transitionMatrix = self.getTransitionMatrix(params)

        # THIS SHOULD NEVER RLLY HAPPEN
        # If no probability vectors are given
        if probabilityVectors == None:
            longTermPrediction = np.zeros(transitionMatrix.shape[1])
            longTermPrediction[0] = 1

        # SHOULD ALWAYS BE THIS
        # If many probability vectors are given
        elif type(probabilityVectors[0]) == tuple:
            actualVector = list(probabilityVectors[0][1].values())
            longTermPrediction = actualVector
            time = probabilityVectors[0][0]
            
        # If only one probability vector is given
        elif type(probabilityVectors[0]) == float:
            longTermPrediction = probabilityVectors
            exit()
        else:
            print("What is probVectors?")
            exit()

        fig = plt.gcf()
        ax = fig.gca()
        ax.cla()

        try:
            # while  or longTermPrediction[len(longTermPrediction)-1] < 0.95:
            for probVector in probabilityVectors:

                if self.plotting == False:
                    print("Plotting var is ", self.plotting)
                    return

                if showLongTermPrediction == True:
                    longTermPrediction = np.matmul(longTermPrediction, transitionMatrix)
                    # Compute loss
                    averageLoss = sum(abs(longTermPrediction[i]-list(probVector[1].values())[i]) for i in range(len(longTermPrediction)))/len(longTermPrediction)
                # print(longTermPrediction)

                ax.cla()

                # If all probability vectors are given, plot them too
                if probabilityVectors != None and type(probabilityVectors[0]) == tuple:
                    
                    if showLongTermPrediction == True:
                        # Uses accumulated vector to predict what the next vector will be
                        ax.plot([i for i in range(len(longTermPrediction))], longTermPrediction, linestyle='--', label="long term prediction, average loss="+str(averageLoss))

                    if showShortTermPrediction == True:
                        # Uses vector from previous loop to compute prediction of next vector
                        shortTermPrediction = np.matmul(actualVector, transitionMatrix)
                        ax.plot([i for i in range(len(shortTermPrediction))], shortTermPrediction, linestyle=':', label="short term prediction")

                    # Plots the actual next vector
                    actualVector = list(probVector[1].values())
                    ax.plot([i for i in range(len(actualVector))], actualVector, label="actual vector")
                    # n += 1

                    ax.set_title(time)

                    time = probVector[0]

                    # print(sum(longTermPrediction), sum(actualVector))

                    # If we have probability vectors to compare the walk to, flush the probability at the end, same as in the loss function
                    # longTermPrediction[len(longTermPrediction)-1] = 0
                    # Also, introduce new probability at the start equal to the amount introduced to the actual probability vectors
                    # longTermPrediction[0] = actualVector[0]

                else:
                    ax.plot([n for n in range(len(longTermPrediction))], longTermPrediction, label="long term prediction")

                fig.legend()
                # if probabilityVectors == None:
                # plt.ylim(-0.01, 0.05)
                plt.ylim(-0.1, 1.1)
                plt.show(block=False)
                plt.pause(0.0001)

        except IndexError:
            pass

        
        plt.close('all')
        self.plotting = False

    def stopPlotting(self):
        print("Set plotting var to False")
        self.plotting = False
#====================================================================================================================

    def viewTransitionMatrixOnPlot(self, transitionMatrix, probVectors, timeRange):


        fig = plt.figure()
        ax = fig.add_subplot(111)

        time = datetime.datetime.combine(datetime.date.min, timeRange[0])
        endTime = datetime.datetime.combine(datetime.date.min, timeRange[1])

        # The vector that the random walker will start on
        # Once the below for loop starts, any probability introduced into the first location in the already-computed prob vectors will also be introduced into the random walker's prob vector
        rwVector = list(probVectors[time.time()].values())

        for t in probVectors:

            probVector = list(probVectors[t].values())

            ax.clear()
            ax.plot([k for k in range(len(probVector))], probVector, label="Real data", alpha=0.5)
            ax.plot([k for k in range(len(rwVector))], rwVector, label="Random walk")

            ax.set_title(t)
            fig.legend()
            plt.show(block=False)
            plt.pause(1)


            rwVector = np.matmul(rwVector, transitionMatrix)

        exit()


class ModelEvaluator():

    def __init__(self, modelParams, testProbVectors, testArrivalTimes):
        self.modelParams = modelParams
        self.testArrivalTimes = testArrivalTimes
        self.testProbVectors = testProbVectors
        # self.departureTime = probabilityVectors[0][0]
        self.rwm = RandomWalkModeller()
        self.pvc = ProbabilityVectorCalculator()

    def computeSpecificProbableArrivalTime(self, location, timeStepInterval, arrivalThreshold, startTime):

        transitionMatrix = self.rwm.getTransitionMatrix(self.modelParams)
        time = datetime.datetime.combine(datetime.date.min, startTime)

        probVector = pd.Series(data=[0]*self.testArrivalTimes.shape[1], index=self.testArrivalTimes.columns)
        probVector[0] = 1
        # print(probVector.loc[location:])
        # exit()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        

        while sum(probVector.loc[location:]) <= 0.95:
            ax.cla()

            probVector = np.matmul(probVector, transitionMatrix)
            probVector = pd.Series(data=probVector, index=self.testArrivalTimes.columns)
            time += timeStepInterval

            ax.plot([str(i) for i in probVector.index], probVector)
            plt.ylim(-0.1, 1.1)
            plt.show(block=False)
            plt.pause(0.001)
        
        return time.time(), probVector[location]


    # def computeAllProbableArrivalTimes(self, timeStepInterval, arrivalThreshold, startTime):

    #     transitionMatrix = self.rwm.getTransitionMatrix(self.modelParams)

    #     probVector = pd.Series(data=[0]*self.testArrivalTimes.shape[1], index=self.testArrivalTimes.columns)
    #     probVector[0] = 1
        
    #     probableArrivals = pd.Series()

    #     time = datetime.datetime.combine(datetime.date.min, startTime)

    #     while probVector.shape[0] > probableArrivals.shape[0]:

    #         for l in probVector.index:
    #             if l in probableArrivals:
    #                 continue

    #             # print(l)

    #             prob = sum(probVector.loc[l:])
    #             if prob >= arrivalThreshold:
    #                 probableArrivals.update({l:time})

    #         probVector = np.matmul(probVector, transitionMatrix)
    #         probVector = pd.Series(data=probVector, index=self.testArrivalTimes.columns)

    #         fig = plt.gcf()
    #         ax=fig.gca()
    #         ax.cla()
    #         ax.plot([str(i) for i in probVector.index], probVector)
    #         plt.show(block=False)
    #         plt.pause(0.001)

    #         # print(time)
    #         time += timeStepInterval

    #     return probableArrivals

    def evaluateModelWithTestSet(self):

        transitionMatrix = self.rwm.getTransitionMatrix(self.modelParams)
        
        # Initialize prob vector
        predictionVector = pd.Series(data=[0]*self.testArrivalTimes.shape[1], index=self.testArrivalTimes.columns)
        predictionVector[0] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Keep going until arrival at last stop is certain
        n = 0
        while predictionVector[-1] < 1:

            averageLoss = sum(abs(predictionVector[i]-list(self.testProbVectors[n][1].values())[i]) for i in range(len(predictionVector)))/len(predictionVector)

            ax.cla()
            ax.plot([s for s in range(len(predictionVector))], predictionVector, label="Average loss:"+str(averageLoss))
            if n < len(self.testProbVectors[n][1].values()):
                ax.plot([s for s in range(len(self.testProbVectors[n][1]))], list(self.testProbVectors[n][1].values()))
            fig.legend()
            plt.ylim(-0.1, 1.1)
            plt.show(block=False)
            plt.pause(0.0001)

            predictionVector = np.matmul(predictionVector, transitionMatrix)

            n += 1
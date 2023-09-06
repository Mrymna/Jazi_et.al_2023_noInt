import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
class NeuronAutopi:
    
    """
    Class developed to analyze the firing of a neuron during the path integration task
    
    It focuses on the analysis of instantaneous firing rate as a function of instantaneous behavioral variables
    
    The behavioral variables come from the NavPath class of the autopipy package
    
    Attributes:
        name: Name of the trial, usually sessionName_trialNo-JourneyNo
        sessionName: Name of the session in which the trial was performed
        trialNo: Trial number within the session
        jouneyNo: Jouney number within the trial
        mousePose: DataFrame with 
        
        ...
    
    Methods:
          
    """
    def __init__(self,name,ifr,navPathSummary,iNavPath,trialElectro):
        """
        Constructor
        
        Arguments:
        name: a name to identify the object
        ifr: tuple containing [0] instantaneous firing rate and [1] the time of each firing rate estimate
        navPathSummary: summary of the navPaths performed during the recordings
        iNavPath: instantaneous behavioral variables
        trialElectro: one object of the autopipy:TrialElectro types. Used to plot the experimental setup
        """
        self.name = name
        self.ifr = ifr
        self.navPathSummary = navPathSummary
        self.iNavPath = iNavPath
        self.trialElectro = trialElectro 
        self.lightConditions = self.navPathSummary.light.unique()
        self.navPathTypes = self.navPathSummary.type.unique()
        self.navPathResultsDict = {}
        self.navPathResultsDict["matrix"]={}
        self.navPathResultsDict["histo"]={}
        
    
    def getSingleNavPathData(self,navPathNames = []):
        """
        Get the IFR and behavioral data for one or more navPaths
        
        Arguments:
        navPathNames: list of navPathNames to select
        """
        
        inav = self.iNavPath[self.iNavPath.name.isin(navPathNames)]
  
        
        if inav.shape[0] == 0:
            print("no inav data with the following navPathNames")
            print(navPathNames)
            print("available navPathNames")
            print(self.iNavPath.name.unique())  
            return None,None
        
        
        ifrIndex= self.isin_tolerance(self.ifr[1],inav.timeRes,tol=0.00001)
        ifr = self.ifr[0][ifrIndex]
        
        if ifr.shape[0] != inav.shape[0]:
            print("problem with shape of ifr {} and inav {} with navPathName {}".format(ifr.shape,inav.shape,navPathNames))
            return None,None
        
        return ifr,inav
    def getNavPathNames(self,navPathType="all",light="light",nLeverMin = 1):
        """
        Get the names of navPaths of a given type
        
        Useful if you want to loop with the navPaths of a given type 
        
        Arguments:
        navPathType: type of navPath to select
        light: light condition.
        nLeverMin: minimum number of lever presses associated with the journey
        
        Returns list of navPaths name
        """
        
        return self.navPathSummary.name[(self.navPathSummary.type==navPathType) & (self.navPathSummary.light==light) & (self.navPathSummary.nLeverPresses >= nLeverMin)]
    
    def navPathResTimeInterval(self,navPathType="all",light="light",nLeverMin = 1):
        """
        Get the res time intervals for a given type of navPaths
        
        This can be used to create a firing rate map out of the data from all navPaths of a given type.
        
        
        Arguments:
        navPathType: type of navPath to select
        light: light condition
        nLeverMin: minimum number of lever presses associated with the journey
        
        Returns numpy array of shape x,2
        
        """
        s = self.navPathSummary.startTimeRes[(self.navPathSummary.type==navPathType) & (self.navPathSummary.light==light) & (self.navPathSummary.nLeverPresses >= nLeverMin)]
        e = self.navPathSummary.endTimeRes[(self.navPathSummary.type==navPathType) & (self.navPathSummary.light==light) & (self.navPathSummary.nLeverPresses >= nLeverMin)]
        return np.vstack([s.to_numpy(),e.to_numpy()]).T
    
    def navPathBehavioralMatrix(self,navPathType="all",light="light",nLeverMin = 1,behavioralVariable = "distance",bins=np.arange(0,70,2), smoothing=True, smoothingSigma = 2):
        """
        Calculate a matrix with the mean firing rate of the neuron during the navPaths as a function 
        of a behavioral variable
        
        The matrix is saved in self.navPathMatrixDict 
        
        Arguments
        navPathType: type of navPaths
        light: light condition, light or dark
        nLeverMin: minimum number of lever presses associated with a journey that the navPath is part of
        behavioralVariable: variable on the x axis
        bins: passed to the stats.binned_statistic function, see documentation in scipy.
        """
        names = self.getNavPathNames(navPathType,light,nLeverMin)
        myList = []
        for n in names:
            ifr,inav = self.getSingleNavPathData(navPathNames=[n])
            if ifr is None or inav is None: # we don't have valid data
                a = np.zeros(bins.shape[0]-1)
                a[:] = np.nan
                myList.append(a)
            else:
                indices = ~np.isnan(inav[behavioralVariable])
                if np.sum(indices)>0: # we have valid behavioral data 
                    res = stats.binned_statistic(inav[behavioralVariable][indices], #binned statistic will get the average of ifr in within bins of behavioral values
                                                 ifr[indices],
                                                 bins=bins)
                    oneArray = res[0].copy()
                    if smoothing:
                        self.smoothRow(oneArray) # will change the values in oneArray, row by row
                        
                    myList.append(oneArray)
                    
                else:
                    a = np.zeros(bins.shape[0]-1)
                    a[:] = np.nan
                    myList.append(a)
                
        m = np.vstack(myList)
        self.navPathResultsDict["matrix"][navPathType+"_"+light+"_"+behavioralVariable] = {"matrix":m,"bins":bins,"names":names}
        
        return
    
    def smoothRow(self, m,smoothingSigma=1):
        """
        Smooth the trial data (1D array) using values from the first valid to last valid to avoid problem with smoothing array containing np.nan
        If there are less than 5 np.nan between the first and last valid value in the array, this will be interpolated and smooth.
        No smoothing will be attempted if there are more than 5 np.nan between the first and last valid values
        """
        start = np.argmax(~np.isnan(m)) # will get the first that is not nan
        end = np.argmax(~np.isnan(np.flip(m))) # index counting from the end
        if end == 0 : # this means the last valid data point was the last data point in the array
            f=m[start:]
            if np.sum(np.isnan(f)) > 0 and np.sum(np.isnan(f)) < 5 :
                f = self.fill_nan(f)
            if np.sum(np.isnan(f)) > 5 :
                #print("smoothRow with {} invalid values, too many missing value, no smoothing applied".format(np.sum(np.isnan(f))))
                return
            m[start:] = gaussian_filter1d(f,sigma=smoothingSigma,mode="nearest")
            
        else :
            f=m[start:-end]
            if np.sum(np.isnan(f)) > 0 and np.sum(np.isnan(f)) < 5 :
                f = self.fill_nan(f)
            if np.sum(np.isnan(f)) > 5 :
                #print("smoothRow with {} invalid values, too many missing value, no smoothing applied".format(np.sum(np.isnan(f))))
                return
            m[start:-end] = gaussian_filter1d(f,sigma=smoothingSigma,mode="nearest")
        
    def fill_nan(self,A):
        """
        interpolate to fill nan values
        """
        inds = np.arange(A.shape[0])
        good = np.where(np.isfinite(A))
        f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
        B = np.where(np.isfinite(A),A,f(inds))
        return B
    
    
    def navPathBehavioralMatrix_targetToAnimalAngle(self,navPathType="all",light="light",nLeverMin = 1,maxTargetDistance = 17.0, bins=np.arange(-np.pi,np.pi,np.pi*2/36)):
        """
        
        Version of navPathBehavioralMatrix adpated to the targetToAnimalAngle, as we want to filter the data to keep only the data when the animal is close to the target
        
        
        Calculate a matrix with the mean firing rate of the neuron during the navPaths as a function 
        of a behavioral variable
        
        The matrix is saved in self.navPathMatrixDict 
        
        Arguments
        navPathType: type of navPaths
        light: light condition, light or dark
        nLeverMin: minimum number of lever presses associated with a journey that the navPath is part of
        maxTargetDistance: set a threshold for data selection, eliminates the data when the animal is far 
        bins: passed to the stats.binned_statistic function, see documentation in scipy.
        """
        behavioralVariable = "targetToAnimalAngle"
        
        names = self.getNavPathNames(navPathType,light,nLeverMin)
        myList = []
        for n in names:# for each NavPath
            ifr,inav = self.getSingleNavPathData(navPathNames=[n])
            if ifr is None or inav is None: # if no valid data
                print("Invalid ifr or inav")
                a = np.zeros(bins.shape[0]-1)
                a[:] = np.nan
                myList.append(a)
            else:
                indices = ~np.isnan(inav[behavioralVariable]) # get valid data, True is kept, False is rejected
                
                # filter for maxTargetDistance, reject when the animal is too far by setting to False
                indices[inav["targetDistance"]>maxTargetDistance] = False
                
                if np.sum(indices)>0: # we have at least one data point
                    res = stats.binned_statistic(inav[behavioralVariable][indices],
                                                 ifr[indices],
                                                 bins=bins)
                    myList.append(res[0])
                else:
                    a = np.zeros(bins.shape[0]-1)
                    a[:] = np.nan
                    myList.append(a)
                
        m = np.vstack(myList)
        self.navPathResultsDict["matrix"][navPathType+"_"+light+"_"+behavioralVariable] = {"matrix":m,"bins":bins}
        
        return
    
    
    
    
    def navPathBehavioralHistogram(self,navPathType="all",light="light",nLeverMin = 1,behavioralVariable = "targetDistance",bins=np.arange(0,40,2)):
        """
        Calculate rate histogram including all navPaths of a given type. A single histogram is created
        
        Arguments
        navPathType: type of navPaths
        light: light condition, light or dark
        nLeverMin: minimum number of lever presses associated with a journey that the navPath is part of
        behavioralVariable: variable on the x axis
        bins: passed to the stats.binned_statistic function, see documentation in scipy.
        """
        names = self.getNavPathNames(navPathType=navPathType,light=light,nLeverMin=nLeverMin)
        naIfr,naInav = self.getSingleNavPathData(navPathNames=names)
        res = stats.binned_statistic(naInav[behavioralVariable],naIfr,bins=bins)
        self.navPathResultsDict["histo"][navPathType+"_"+light+"_"+behavioralVariable] = {"histo":res[0],"bins":res[1]}
    
    
    
    def plotNavPathBehavioralMatrix(self,ax, navPathType="all",light="light",behavioralVariable = "distance",title=""):
        """
        Plot the matrix with the mean firing rate of the neuron during the navPaths as a function of a behavioral variable
        
        The matrix is generated by self.navPathBehavioralMatrix() and stored in self.navPathMatrixDict
        
        Arguments
        ax: plot axis on which to plot
        navPathType: type of navPaths
        light: light condition, light or dark
        behavioralVariable: variable on the x axis
        """
        if  navPathType+"_"+light+"_"+behavioralVariable not in self.navPathResultsDict["matrix"]:
            print("matrix "+ navPathType+"_"+light+"_"+behavioralVariable  + " not in the self.navPathResultsDict['matrix']")
        
        m = self.navPathMatrixDict[navPathType+"_"+light+"_"+behavioralVariable]["matrix"]
        bins = self.navPathMatrixDict[navPathType+"_"+light+"_"+behavioralVariable]["bins"]
        
        ax.imshow(m,aspect="auto",interpolation="none",extent=[np.min(bins),np.max(bins),0,m.shape[0]],origin="lower",cmap="jet")
        
        ax.set_ylabel("Paths")
        ax.set_xlabel("{}".format(behavioralVariable.capitalize()))
        ax.set_title("{} {:.2f} Hz".format(title,np.nanmax(m)))
        
    def plotNavPathBehavioralMatrixMean(self,ax, navPathType="all",light="light",behavioralVariable = "distance",title=""):
        """
        Plot the mean rate of the matrix with the mean firing rate of the neuron during the navPaths as a function of a behavioral variable
        
        The matrix is generated by self.navPathBehavioralMatrix() and stored in self.navPathMatrixDict
        
        Arguments
        ax: plot axis on which to plot
        navPathType: type of navPaths
        light: light condition, light or dark
        behavioralVariable: variable on the x axis
        """
        
        if  navPathType+"_"+light+"_"+behavioralVariable not in self.navPathResultsDict["matrix"]:
            print("matrix "+ navPathType+"_"+light+"_"+behavioralVariable  + " not in the self.navPathResultsDict['matrix']")
        
        m = self.navPathResultsDict["matrix"][navPathType+"_"+light+"_"+behavioralVariable]["matrix"]
        bins = self.navPathResultsDict["matrix"][navPathType+"_"+light+"_"+behavioralVariable]["bins"]
        stepSize = bins[1]-bins[0]
        x = bins[:-1]+stepSize/2
        
        M = np.nanmean(m,axis=0)
        ax.plot(x,M)
        if np.nanmax(M) > 0:
            ax.set_ylim(0,np.nanmax(M))
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("{}".format(behavioralVariable.capitalize()))
        ax.set_title("{}".format(title))
    
    def plotNavPath(self, ax, navPathType = "all", light="light",title="",xlabel="x position (cm)",ylabel="y position (cm)"):
        """
        plot the animal path for a combination of navPath type and light condition
        
        Arguments
        ax : plot axis
        navPathType: type of NavPath to plot
        light: light condition
        """
        selNavPaths = self.navPathSummary.name[(self.navPathSummary.type==navPathType) & (self.navPathSummary.light==light)]
        inav_sel = self.iNavPath[self.iNavPath.name.isin(selNavPaths)]
        
        self.trialElectro.plotTrialSetup(ax=ax,title = "", bridge=True,homeBase=True,lever=False)
        
        
        for name in selNavPaths:
            df = self.iNavPath[self.iNavPath.name == name]
            ax.plot(df.x,df.y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def plotAllNavPaths(self):
        """
        plot the animal path for all combinations of navPath types and light conditions
        This generates a figure with several axes
        
        """
        
        cols=len(self.navPathTypes)
        rows=len(self.lightConditions)
        
        fig, axes = plt.subplots(rows,cols,figsize=(cols*4,rows*5))
        for i,l in enumerate(self.lightConditions):
            for j,p in enumerate(self.navPathTypes):
                ax= axes[i,j]
                self.plotNavPath(ax,navPathType=p,light=l)

        plt.show()
        
        
        
    def __str__(self):
        return  str(self.__class__) + '\n' + '\n'.join((str(item) + ' = ' + str(self.__dict__[item]) for item in self.__dict__))
    
    
    def isin_tolerance(self, A, B, tol):
        """
        Are elements of A in B, with some tolerance

        """
        A = np.asarray(A)
        B = np.asarray(B)

        if B.size<2:
            print("B should have a size of at least 2")

        Bs = np.sort(B) # skip if already sorted
        idx = np.searchsorted(Bs, A)

        linvalid_mask = idx==len(B)
        idx[linvalid_mask] = len(B)-1
        lval = Bs[idx] - A
        lval[linvalid_mask] *=-1

        rinvalid_mask = idx==0
        idx1 = idx-1
        idx1[rinvalid_mask] = 0
        rval = A - Bs[idx1]
        rval[rinvalid_mask] *=-1
        return np.minimum(lval, rval) <= tol
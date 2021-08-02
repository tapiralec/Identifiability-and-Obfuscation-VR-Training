import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from agm_util.dataloader import ToXy
from os import path
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

classifier = {
        'kNN': lambda x: KNeighborsClassifier(n_neighbors=x[0]),
        'RF': lambda x: RandomForestClassifier(),
        'GBM': lambda x: GradientBoostingClassifier(),
        'gbm2': lambda x: GradientBoostingClassifier(n_estimators=20,max_depth=10,min_samples_leaf=10)
        }

def FindCache(predFile):
    return path.join(GetCacheDir(),predFile)

def GetCacheDir():
    return(path.join(path.dirname(__file__),'predcache'))

def Normalize(dftrain,dfpred):
    LX,Ly = ToXy(dftrain)
    Lid = dftrain[['subphaseID','PID','PIDname']]
    Lcols = dftrain.drop(['subphaseID','PID','PIDname'],axis=1).columns
    PX,Py = ToXy(dfpred)
    Pid = dftrain[['subphaseID','PID','PIDname']]
    Pcols = dfpred.drop(['subphaseID','PID','PIDname'],axis=1).columns
    scaler = StandardScaler()
    scaler.fit(LX)
    #transform both X matrices
    nLX = scaler.transform(LX)
    nPX = scaler.transform(PX)
    #turn back into nice dataframes
    Lout = pd.DataFrame(nLX,columns=Lcols,index=dftrain.index.copy())
    Pout = pd.DataFrame(nPX,columns=Pcols,index=dfpred.index.copy())
    Lout['subphaseID'] = Lid['subphaseID']
    Lout['PID'] = Lid['PID']
    Lout['PIDname'] = Lid['PIDname']
    Pout['subphaseID'] = dfpred['subphaseID']
    Pout['PID'] = dfpred['PID']
    Pout['PIDname'] = dfpred['PIDname']

    return Lout,Pout

def NormalizeList(dfpairlist):
    for dfpair in dfpairlist:
        yield Normalize(dfpair[0],dfpair[1])


def MakeClassifier(cls,df,hyperpara=None):
    X, y = ToXy(df)
    clf = classifier[cls](hyperpara)
    clf.fit(X,y)
    return clf

def LoadPrediction(predFile):
    if path.exists(f := FindCache(predFile)):
        return pd.read_csv(f,parse_dates=['Timestamp'],index_col=['Timestamp'])

def GetOrLoadPrediction(clf,df,predFile='',overrideprev=False):
    ''' gets or loads a prediction on a given classifier '''
    df = df.copy()
    writeout = False
    if not overrideprev and predFile!='':
        if path.exists(f := FindCache(predFile)):
            return pd.read_csv(f,parse_dates=['Timestamp'],index_col=['Timestamp'])
            #df = pd.DataFrame()
            #df = pickle.load(f,'rb')
        else:
            writeout = True
    df['Prediction'] = clf.predict(ToXy(df)[0])
    if (writeout or overrideprev) and predFile!='':
        df.to_csv(FindCache(predFile))
    return df

#def EvaluatePredictionByPhase(df,PIDcol='PID',predcol='Prediction',phasecol = 'subphaseID'):
#    df['
def EvaluatePrediction(df,groups=['PID','subphaseID'],predcol='Prediction'):
    ret = pd.DataFrame(df.groupby(groups)[predcol].apply(lambda x: x.value_counts(dropna=False).idxmax()))
    if len(groups)>1:
        ret['Correct'] = ret.apply(lambda x: x.name[0]==x[predcol],axis=1)
    else:
        ret['Correct'] = ret.apply(lambda x: x.name==x[predcol],axis=1)
    return ret

#for knn in range(10):
    #for train,pred in inoutpairs:
        #BatteryTest('knn',knn,train,pred)

def BatteryTest(cls,knn,posvel,trainpreddfs):
    i,(dftrain,dfpred) = trainpreddfs
    print(f'Battery testing, {cls} {knn} {i}')
    clf=MakeClassifier(cls,dftrain,hyperpara=knn)
    GetOrLoadPrediction(clf,dfpred,f'cls{cls}_knn{knn}_i{i}_{posvel}.csv')


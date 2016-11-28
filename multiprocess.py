import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Process
import os
event_train=pd.DataFrame.from_csv("events_train.tsv",sep="\t",header=0)
event_train_s=event_train[["closed_tstamp","event_type","latitude","longitude"]]
prediction_trials=pd.DataFrame.from_csv("prediction_trials.tsv",sep="\t",header=0)
prediction_trials['start_dt']=pd.to_datetime(prediction_trials.apply(lambda x:x['start'][:-6],axis=1),format='%Y-%m-%dT%H:%M:%S',errors='coerce' )
prediction_trials['end_dt']=pd.to_datetime(prediction_trials.apply(lambda x:x['end'][:-6],axis=1),format='%Y-%m-%dT%H:%M:%S',errors='coerce' )
event_train_s=event_train_s[~event_train_s['closed_tstamp'].isnull()]
event_train_s['dt']=pd.to_datetime(event_train_s.apply(lambda x:x['closed_tstamp'][:-6],axis=1),format='%Y-%m-%dT%H:%M:%S',errors='coerce' )
event_train_s=event_train_s[event_train_s.dt<'2013-01-01']

A_event=event_train_s[(event_train_s['event_type']=='accidentsAndIncidents')]
R_event=event_train_s[(event_train_s['event_type']=='roadwork')]
P_event=event_train_s[(event_train_s['event_type']=='precipitation')]
D_event=event_train_s[(event_train_s['event_type']=='deviceStatus')]
O_event=event_train_s[(event_train_s['event_type']=='obstruction')]
T_event=event_train_s[(event_train_s['event_type']=='trafficConditions')]

def pred_score(event,start,end):
    size=end-start
    regressor_quadratic_score=np.array([0.0]*size)
    regressor_quadratic_pred=np.array([0.0]*size)
    model_score=np.array([0.0]*size)
    model_pred=np.array([0.0]*size)
    for i in range(size):
        test=prediction_trials.iloc[i+start]
        test_cnt=event[(event.latitude<=test.nw_lat) & (event.latitude>=test.se_lat) & (event.longitude>=test.se_lon) & (event.longitude>=test.nw_lon)]
        if(len(test_cnt)==0):
            model_pred[i]=0
            model_score[i]=0
            regressor_quadratic_score[i]=0
            regressor_quadratic_pred[i]=0
        else:
            t=test_cnt.groupby(test_cnt.dt.dt.year).size()
            X=np.array(t.index.astype(int)).reshape(len(t), 1)
            y=np.array(t.values)
            model = LinearRegression()
            model.fit(X,y)
            model_pred[i]=model.predict(2014)
            model_score[i]=model.score(X,y)
            quadratic_featurizer = PolynomialFeatures(degree=2)
            X_train_quadratic=quadratic_featurizer.fit_transform(X)
            model.fit(X_train_quadratic, y)
            regressor_quadratic_score[i]=model.score(X_train_quadratic,y)
            regressor_quadratic_pred[i]=model.predict(quadratic_featurizer.transform(np.array([2014]).reshape(-1, 1)))
    np.savetxt("tmp/quadratic_score_"+str(start)+'_'+str(end)+'.txt',regressor_quadratic_score,fmt='%.10f')
    np.savetxt("tmp/model_score_"+str(start)+'_'+str(end)+'.txt',model_score,fmt='%.10f')
    np.savetxt("tmp/quadratic_pred_"+str(start)+'_'+str(end)+'.txt',regressor_quadratic_pred,fmt='%.10f')
    np.savetxt("tmp/model_pred_"+str(start)+'_'+str(end)+'.txt',model_pred,fmt='%.10f')



if __name__ == '__main__':
	divide=3
	event='T_event'
	os.mkdir(event)
	start=range(0,len(prediction_trials),len(prediction_trials)/divide)
	end=start[1:]
	end.append(len(prediction_trials))
	subprocess=[]
	for i in range(divide):
		p=Process(target=pred_score, args=(T_event,start[i],end[i],))
		p.start()
		subprocess.append(p)
	while subprocess:
		subprocess.pop().join()

	quadratic_score=np.array([])
	model_score=np.array([])
	quadratic_pred=np.array([])
	model_pred=np.array([])	
	for i in range(divide):
    		quadratic_score=np.append(quadratic_score,np.loadtxt("tmp/quadratic_score_"+str(start[i])+'_'+str(end[i])+'.txt'))
		quadratic_pred=np.append(quadratic_pred,np.loadtxt("tmp/quadratic_pred_"+str(start[i])+'_'+str(end[i])+'.txt'))
		model_score=np.append(model_score,np.loadtxt("tmp/model_score_"+str(start[i])+'_'+str(end[i])+'.txt'))
		model_pred=np.append(model_pred,np.loadtxt("tmp/model_pred_"+str(start[i])+'_'+str(end[i])+'.txt'))
	np.savetxt(event+"/quadratic_score.txt",quadratic_score,fmt='%.10f')
	np.savetxt(event+"/model_score.txt",model_score,fmt='%.10f')
	np.savetxt(event+"/quadratic_pred.txt",quadratic_pred,fmt='%.10f')	
	np.savetxt(event+"/model_pred.txt",model_pred,fmt='%.10f')	
		

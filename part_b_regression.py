import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv'); X=df[FEATURES]; y=df.delivery_hours
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42)
sc=StandardScaler().fit(X_train); tr=sc.transform(X_train); te=sc.transform(X_test)
pd.DataFrame([('train_rows',len(X_train)),('val_rows',len(X_val)),('test_rows',len(X_test)),('train_scaled_max_abs_mean',np.abs(tr.mean(axis=0)).max()),('test_scaled_max_abs_mean',np.abs(te.mean(axis=0)).max())],columns=['metric','value']).to_csv('outputs/part_b_checks.csv',index=False)
def metrics(name,split,yt,yp):
    mse=mean_squared_error(yt,yp); return [name,split,mean_absolute_error(yt,yp),mse,np.sqrt(mse),r2_score(yt,yp)]
rows=[metrics('mean_baseline','val',y_val,np.repeat(y_train.mean(),len(y_val)))]
depths=[1,2,3,4,6,8,12,None]; tree_rows=[]
for d in depths:
    model=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train)
    tree_rows.append(['unlimited' if d is None else d,np.sqrt(mean_squared_error(y_train,model.predict(X_train))),np.sqrt(mean_squared_error(y_val,model.predict(X_val)))])
pd.DataFrame(tree_rows,columns=['max_depth','train_rmse','val_rmse']).round(4).to_csv('outputs/part_b_tree_depth.csv',index=False)
best_depth=min(tree_rows,key=lambda r:(r[2], 999 if r[0]=='unlimited' else r[0]))[0]
if best_depth=='unlimited': best_depth=None
models={
'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
'decision_tree':DecisionTreeRegressor(max_depth=best_depth,random_state=42),
'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),
'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42)}
for name,m in models.items():
    m.fit(X_train,y_train); rows.append(metrics(name,'val',y_val,m.predict(X_val)))
pd.DataFrame(rows,columns=['model','split','MAE','MSE','RMSE','R2']).round(4).to_csv('outputs/part_b_metrics.csv',index=False)
gb=[]
for lr in [.01,.1,1.0]:
    m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
    yp=m.predict(X_val); gb.append([lr,np.sqrt(mean_squared_error(y_val,yp)),r2_score(y_val,yp)])
pd.DataFrame(gb,columns=['learning_rate','val_rmse','val_r2']).round(4).to_csv('outputs/part_b_gb_learning_rate.csv',index=False)
rank=sorted([(r[0],r[4]) for r in rows[1:]],key=lambda x:x[1]); top=[x[0] for x in rank[:2]]
X_tv=pd.concat([X_train,X_val]); y_tv=pd.concat([y_train,y_val]); kf=KFold(n_splits=5,shuffle=True,random_state=42)
cvrows=[]
for name in top:
    m=models[name]
    scores=cross_val_score(m,X_tv,y_tv,cv=kf)
    cvrows.append([name,scores.mean(),scores.std()])
pd.DataFrame(cvrows,columns=['model','mean_r2','std_r2']).round(4).to_csv('outputs/part_b_cv.csv',index=False)
winner=rank[0][0]; final=models[winner]
pd.DataFrame([metrics(winner,'test',y_test,final.predict(X_test))],columns=['model','split','MAE','MSE','RMSE','R2']).round(4).to_csv('outputs/part_b_test_tmp.csv',index=False)
allm=pd.read_csv('outputs/part_b_metrics.csv'); allm=pd.concat([allm,pd.read_csv('outputs/part_b_test_tmp.csv')],ignore_index=True); allm.to_csv('outputs/part_b_metrics.csv',index=False); os.remove('outputs/part_b_test_tmp.csv')
print('Part B complete; winner:',winner,'depth:',best_depth)

import os
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES,clean_shipments
os.makedirs('models',exist_ok=True); os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42,stratify=y_tmp)
conf=pd.read_csv('outputs/part_c_final_config.csv').iloc[0]
md=conf.max_depth; md=None if str(md)=='unlimited' or pd.isna(md) else int(md)
model=Pipeline([('selector',SelectKBest(f_classif,k=int(conf.selector_k))),('model',RandomForestClassifier(n_estimators=int(conf.n_estimators),max_depth=md,min_samples_leaf=int(conf.min_samples_leaf),random_state=42))]).fit(X_train,y_train)
joblib.dump(model,'models/delay_model.pkl')
pd.DataFrame({'threshold':[float(conf.threshold)]}).to_csv('models/delay_threshold.csv',index=False)
pd.DataFrame({'feature':FEATURES}).to_csv('models/feature_order.csv',index=False)
raw=pd.read_csv('data/raw_shipments.csv'); _,med=clean_shipments(raw)
pd.DataFrame([{'column':k,'median':v} for k,v in med.items()]).to_csv('models/impute_values.csv',index=False)
loaded=joblib.load('models/delay_model.pkl'); diff=np.max(np.abs(model.predict_proba(X_test)[:,1]-loaded.predict_proba(X_test)[:,1]))
pd.DataFrame([('test_rows_scored',len(X_test)),('reload_max_abs_probability_diff',diff),('threshold',float(conf.threshold))],columns=['metric','value']).to_csv('outputs/part_d_reload_check.csv',index=False)
print('Model packaged; reload diff:',diff)

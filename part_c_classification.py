import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS
os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
X_train,X_tmp,y_train,y_tmp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_tmp,y_tmp,test_size=.5,random_state=42,stratify=y_tmp)
def scores(ytrue,ypred):
    tn,fp,fn,tp=confusion_matrix(ytrue,ypred,labels=[0,1]).ravel()
    return [tn,fp,fn,tp,accuracy_score(ytrue,ypred),precision_score(ytrue,ypred,zero_division=0),recall_score(ytrue,ypred,zero_division=0),f1_score(ytrue,ypred,zero_division=0)]
rows=[]; rows.append(['blind_all_on_time',*scores(y_val,np.zeros(len(y_val),dtype=int))])
log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())]).fit(X_train,y_train); rows.append(['logistic_regression',*scores(y_val,log.predict(X_val))])
tree_cands=[]
for d in [2,3,4,5,6,8,10]:
    m=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(X_train,y_train)
    f=f1_score(y_val,m.predict(X_val),zero_division=0); tree_cands.append((f,d,m))
_,td,tm=sorted(tree_cands,key=lambda z:(-z[0],z[1]))[0]; rows.append([f'decision_tree_depth_{td}',*scores(y_val,tm.predict(X_val))])
rf17=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train); rows.append(['random_forest_default',*scores(y_val,rf17.predict(X_val))])
pd.DataFrame(rows,columns=['model','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4).to_csv('outputs/part_c_models.csv',index=False)
fvals,_=f_classif(X_train,y_train); ranks=pd.Series(fvals,index=FEATURES).rank(ascending=False,method='min').astype(int)
fs=pd.DataFrame({'feature':FEATURES,'f_score':fvals,'f_rank':ranks.values,'rf_importance':rf17.feature_importances_,'is_noise':[f in NOISE_COLUMNS for f in FEATURES]})
fs=fs.sort_values('f_rank'); fs.round(4).to_csv('outputs/part_c_feature_scores.csv',index=False)
selectrows=[]
for k in [5,7,10,12,17]:
    p=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
    selectrows.append([k,f1_score(y_val,p.predict(X_val),zero_division=0)])
pd.DataFrame(selectrows,columns=['k','val_f1']).round(4).to_csv('outputs/part_c_selectk.csv',index=False)
selector=SelectKBest(f_classif,k=7).fit(X_train,y_train); Xtr=selector.transform(X_train); Xv=selector.transform(X_val)
param_grid={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5).fit(Xtr,y_train)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),param_grid,n_iter=9,cv=5,random_state=42).fit(Xtr,y_train)
def val_f1(est): return f1_score(y_val,est.predict(Xv),zero_division=0)
base=RandomForestClassifier(n_estimators=100,random_state=42,min_samples_leaf=1).fit(Xtr,y_train)
tuning=[['grid_search',27,5,135,json.dumps(grid.best_params_,sort_keys=True),grid.best_score_,val_f1(grid.best_estimator_)],['randomized_search',9,5,45,json.dumps(rand.best_params_,sort_keys=True),rand.best_score_,val_f1(rand.best_estimator_)],['default_forest_same_7_features',1,0,1,'defaults',np.nan,val_f1(base)]]
pd.DataFrame(tuning,columns=['search','candidates','cv_folds','total_fits','best_params','best_cv_accuracy','val_f1']).round(4).to_csv('outputs/part_c_tuning.csv',index=False)
tdf=pd.DataFrame(tuning,columns=['search','candidates','cv_folds','total_fits','best_params','best_cv_accuracy','val_f1'])
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen=sorted(tuning,key=lambda r:(-r[6],priority[r[0]]))[0]
params={'default_forest_same_7_features':{'n_estimators':100,'max_depth':None,'min_samples_leaf':1}}
if chosen[0]=='grid_search': params[chosen[0]]=grid.best_params_
elif chosen[0]=='randomized_search': params[chosen[0]]=rand.best_params_
final_params=params[chosen[0]]
final=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**final_params))]).fit(X_train,y_train)
probs=final.predict_proba(X_val)[:,1]; threshold_rows=[]
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: threshold_rows.append([t,*scores(y_val,(probs>=t).astype(int))])
thdf=pd.DataFrame(threshold_rows,columns=['threshold','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4); thdf.to_csv('outputs/part_c_thresholds.csv',index=False)
eligible=thdf[thdf.recall>=.80]
threshold=float(eligible.threshold.max()) if len(eligible) else .1
ptest=final.predict_proba(X_test)[:,1]; finaltest=pd.DataFrame([[threshold,*scores(y_test,(ptest>=threshold).astype(int))]],columns=['threshold','TN','FP','FN','TP','accuracy','precision','recall','f1']).round(4)
finaltest.to_csv('outputs/part_c_final_test.csv',index=False)
md=final_params.get('max_depth',None)
pd.DataFrame([[7,final_params['n_estimators'],'unlimited' if md is None else md,final_params['min_samples_leaf'],threshold,chosen[0]]],columns=['selector_k','n_estimators','max_depth','min_samples_leaf','threshold','chosen_from']).to_csv('outputs/part_c_final_config.csv',index=False)
print('Part C complete; chosen:',chosen[0],'threshold:',threshold)

import pandas as pd, numpy as np, joblib
from src.shipment_prep import clean_shipments,build_features
raw=pd.read_csv('data/new_shipments.csv'); hubs=pd.read_csv('data/hubs.csv')
meddf=pd.read_csv('models/impute_values.csv'); med=dict(zip(meddf.column,meddf['median']))
clean,_=clean_shipments(raw,medians=med); feat=build_features(clean,hubs)
order=pd.read_csv('models/feature_order.csv').feature.tolist(); model=joblib.load('models/delay_model.pkl'); threshold=float(pd.read_csv('models/delay_threshold.csv').threshold.iloc[0])
feat['delay_probability']=model.predict_proba(feat[order])[:,1].round(4); feat['flagged']=(feat.delay_probability>=threshold).astype(int)
segments=pd.read_csv('outputs/hub_segments.csv')[['hub_id','segment']]; feat=feat.merge(segments,on='hub_id',how='left',validate='many_to_one')
feat['action']=np.where(feat.flagged.eq(0),'No action',np.where(feat.segment.eq('Escalate'),'Escalate to hub manager','Proactive customer call'))
feat[['shipment_id','hub_id','segment','delay_probability','flagged','action']].to_csv('outputs/new_shipment_actions.csv',index=False)
print('Scored',len(feat),'new shipments')

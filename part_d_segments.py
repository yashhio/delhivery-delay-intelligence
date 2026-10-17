import os,itertools
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv')
prof=df.groupby('hub_id',as_index=False).agg(shipments=('shipment_id','count'),mean_delivery_hours=('delivery_hours','mean'),delay_rate=('is_delayed','mean'),mean_distance_km=('distance_km','mean'),mean_n_stops=('n_stops','mean')).sort_values('hub_id').reset_index(drop=True)
prof.round(4).to_csv('outputs/hub_profile.csv',index=False)
cols=['mean_delivery_hours','delay_rate','mean_distance_km','mean_n_stops']; Z=StandardScaler().fit_transform(prof[cols])
sep=[]
for k in [2,3,4,5]:
    labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Z)
    across=[]; within=[]
    for i,j in itertools.combinations(range(len(prof)),2):
        dist=float(np.linalg.norm(Z[i]-Z[j]))
        (within if labels[i]==labels[j] else across).append(dist)
    sizes=sorted(pd.Series(labels).value_counts().tolist(),reverse=True)
    sep.append([k,np.mean(across)/np.mean(within),'-'.join(map(str,sizes))])
pd.DataFrame(sep,columns=['K','across_over_within','cluster_sizes']).assign(across_over_within=lambda x:x.across_over_within.round(4)).to_csv('outputs/part_d_separation.csv',index=False)
labels=KMeans(n_clusters=3,random_state=42,n_init=10).fit_predict(Z); prof['cluster']=labels
order=prof.groupby('cluster').delay_rate.mean().sort_values().index.tolist(); names={order[0]:'Standard',order[1]:'Watch',order[2]:'Escalate'}
prof['segment']=prof.cluster.map(names)
hubs=pd.read_csv('data/hubs.csv')
seg=prof[['hub_id','segment']].merge(hubs[['hub_id','hub_city','hub_tier']],on='hub_id',how='left').sort_values('hub_id')
seg.to_csv('outputs/hub_segments.csv',index=False)
segment_profile=prof.groupby('segment').agg(hubs=('hub_id','count'),mean_delivery_hours=('mean_delivery_hours','mean'),delay_rate=('delay_rate','mean'),mean_distance_km=('mean_distance_km','mean'),mean_n_stops=('mean_n_stops','mean')).reset_index()
segment_profile['order']=segment_profile.segment.map({'Standard':0,'Watch':1,'Escalate':2}); segment_profile=segment_profile.sort_values('order').drop(columns='order')
segment_profile.round(4).to_csv('outputs/segment_profile.csv',index=False)
expected={1:'Standard',2:'Watch',3:'Escalate'}
seg[seg.apply(lambda r:r.segment!=expected[r.hub_tier],axis=1)].to_csv('outputs/tier_vs_segment_mismatch.csv',index=False)
print('Part D segments complete')

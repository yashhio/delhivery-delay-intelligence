import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS
os.makedirs('outputs', exist_ok=True)
raw = pd.read_csv('data/raw_shipments.csv')
hubs = pd.read_csv('data/hubs.csv')
raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()
spell_before = int(dedup['service_type'].nunique())
missing_distance = int(dedup['distance_km'].isna().sum())
missing_weight = int(dedup['parcel_weight_kg'].isna().sum())
missing_value = int(dedup['declared_value'].isna().sum())
missing_payment = int(dedup['payment_mode'].isna().sum())
sentinel = int((dedup['parcel_weight_kg'] > 100).sum())
clean, medians = clean_shipments(raw)
spell_after = int(clean['service_type'].nunique())
feat = build_features(clean, hubs)
feat['is_delayed'] = (feat['delivery_hours'] > feat['promised_hours']).astype(int)
feat.to_csv('data/modelling_table.csv', index=False)
null_count = int(feat[FEATURES + ['delivery_hours','is_delayed']].isna().sum().sum())
q1, q3 = feat['delivery_hours'].quantile([.25,.75]); iqr=q3-q1
lo, hi = q1-1.5*iqr, q3+1.5*iqr
logv=np.log(feat['declared_value_inr'])
checks = [
('raw_rows',raw_rows),('duplicate_rows_removed',raw_rows-len(dedup)),('rows_after_dedup',len(dedup)),('missing_distance_km_before',missing_distance),('missing_parcel_weight_before',missing_weight),('sentinel_weight_rows',sentinel),('missing_declared_value_before',missing_value),('missing_payment_mode_before',missing_payment),('service_type_spellings_before',spell_before),('service_type_spellings_after',spell_after),('median_distance_km',medians['distance_km']),('median_parcel_weight_kg',medians['parcel_weight_kg']),('median_declared_value_inr',medians['declared_value_inr']),('rows_after_hub_merge',len(feat)),('nulls_in_features_and_target',null_count),('delayed_count',int(feat.is_delayed.sum())),('delay_rate',float(feat.is_delayed.mean())),('hours_q1',q1),('hours_q3',q3),('hours_lower_bound',lo),('hours_upper_bound',hi),('hours_outlier_count',int(((feat.delivery_hours<lo)|(feat.delivery_hours>hi)).sum())),('declared_value_mean_inr',feat.declared_value_inr.mean()),('declared_value_median_inr',feat.declared_value_inr.median()),('log_declared_value_mean',logv.mean()),('log_declared_value_median',logv.median())]
checks_df=pd.DataFrame(checks,columns=['metric','value']); int_metrics=['raw_rows','duplicate_rows_removed','rows_after_dedup','missing_distance_km_before','missing_parcel_weight_before','sentinel_weight_rows','missing_declared_value_before','missing_payment_mode_before','service_type_spellings_before','service_type_spellings_after','rows_after_hub_merge','nulls_in_features_and_target','delayed_count','hours_outlier_count']; checks_df['value']=[int(v) if m in int_metrics else round(float(v),4) for m,v in zip(checks_df.metric,checks_df.value)]; checks_df.to_csv('outputs/part_a_checks.csv',index=False)
rows=[]
for dim, series in [('service_type',feat.service_type),('hub_tier',feat.hub_tier),('distance_band',pd.cut(feat.distance_km,bins=[0,200,400,600,2000],labels=['0-200','201-400','401-600','601+'],right=True,include_lowest=True))]:
    temp=feat.assign(_group=series)
    for group,g in temp.groupby('_group',observed=False,dropna=False):
        if pd.isna(group): continue
        rows.append((dim,group,len(g),int(g.is_delayed.sum()),g.is_delayed.mean(),g.delivery_hours.mean()))
pd.DataFrame(rows,columns=['dimension','group','shipments','delayed','delay_rate','mean_hours']).assign(delay_rate=lambda x:x.delay_rate.round(4),mean_hours=lambda x:x.mean_hours.round(4)).to_csv('outputs/part_a_delay_by_group.csv',index=False)
feat.select_dtypes(include=[np.number]).corr().to_csv('outputs/part_a_correlations.csv',index_label='')
print('Part A complete', len(feat), 'rows')

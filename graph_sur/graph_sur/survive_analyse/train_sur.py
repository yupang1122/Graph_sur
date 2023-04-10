import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance

from collections import Counter

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

set_config(display="text")  # displays text representation of estimators

text_path = './TCGA_MEASUREMENTS.xlsx'
text = pd.read_excel(text_path)
survive_list = ['vital_statue','days_to_event']
survive_list_end = list(text['vital_status'] == 1)
survive_list_time = list(text['days_to_event'])

y = np.ndarray([338],dtype = [('vital_status',bool),('days_to_event',float)])
# for i in zip(survive_list_end,survive_list_time)]):
for i in range(len(y)):
    y[i] = (survive_list_end[i],survive_list_time[i])

msi_list = ['nan','NA','MSS','MSI-L','MSI-H']
tumor_site_list = ['NA','colon','rectum','nan']
pathologic_stage_list = ['nan','NA','stage i','stage ii','stage iia','stage iib','stage iic','stage iii','stage iiia','stage iiib','stage iiic','stage iv','stage iva','stage ivb']
pathology_T_stage_list = ['nan','t1','t2','t3','t4','t4a','t4b']
pathology_N_stage_list = ['nan','n0','n1','n1a','n1b','n1c','n2','n2a','n2b','nx']
pathology_M_stage_list = ['nan','m0','m1','m1a','m1b','mx','NA']
gender_list = ['nan','male','female']

x = text.drop(columns=['ID','ADI','BACK','DEB','LYM','MUC','MUS','NORM','STR','TUM','vital_status','tumor_tissue_site','days_to_event','percent_stromal_cells','histological_type','methylation_subtype','RF_predictedCMS','cleanstage'],axis=1)
for i in range(len(x['msi_status'])):
    # x['msi_status'][i] = msi_list.index(x['msi_status'][i])
    x['msi_status'][i] = msi_list.index(str(x['msi_status'][i]))
    # x['tumor_tissue_site'][i] = tumor_site_list.index(str(x['tumor_tissue_site'][i]))
    x['pathologic_stage'][i] = pathologic_stage_list.index(str(x['pathologic_stage'][i]))
    x['pathology_T_stage'][i] = pathology_T_stage_list.index(str(x['pathology_T_stage'][i]))
    x['pathology_N_stage'][i] = pathology_N_stage_list.index(str(x['pathology_N_stage'][i]))
    x['pathology_M_stage'][i] = pathology_M_stage_list.index(str(x['pathology_M_stage'][i]))
    x['gender'][i] = gender_list.index(str(x['gender'][i]))

random_state = 20

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = random_state)

##training

rsf = RandomSurvivalForest(n_estimators=1000,min_samples_split=10,min_samples_leaf=15,n_jobs=1,random_state = random_state)
cox = CoxPHSurvivalAnalysis()
rsf.fit(X_train,y_train)
cox.fit(X_train, y_train)
print('rsf score')
print(rsf.score(X_test,y_test))
print('cox score')
print(cox.score(X_test,y_test))

print('rsf predict hazard')
print(pd.Series(rsf.predict(X_test[:15])))
print('cox predict hazard')
print(pd.Series(cox.predict(X_test[:15])))



####
print('rsf predict importance score')
result = permutation_importance(rsf,X_test,y_test,n_repeats=15,random_state=random_state)

print(pd.DataFrame(
    {k:result[k] for k in ("importances_mean", "importances_std")},
    index = X_test.columns
).sort_values(by="importances_mean",ascending = False)
)
######


#cox analyse hazard factors
print(' cox predict importance score')
estimator = CoxPHSurvivalAnalysis()
estimator.fit(X_test, y_test)
print(pd.Series(estimator.coef_,index=X_test.columns))
####
print('predict survival probability')
rsf_pred_surv = rsf.predict_survival_function(X_test[:5])
cox_pred_surv = estimator.predict_survival_function(X_test[:5])

time_points = np.arange(1,2475)

plt.figure(1)
for i, surv_func in enumerate(rsf_pred_surv):
    plt.step(time_points, surv_func(time_points), where="post",
             label=" rsf Sample %d" % (i + 1))
for i , surv_func in enumerate(cox_pred_surv):
    plt.step(time_points, surv_func(time_points), where = "post",
             label = "cox sample %d" % (i+1))

survive_time = list(text['days_to_event'])

for i in survive_time:
    if i == 0:
        survive_time.remove(i)
times = Counter(survive_time)
single_time = sorted(times)
n = len(survive_time)
x = single_time
y = []
flag = 0
for time in single_time:
    y.append(1 - ((times[time] + flag) / n))
    flag += times[time]
plt.step(x,y,'b',label = 'target')

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")


plt.figure(2)
rsf_surv = rsf.predict_cumulative_hazard_function(X_test[:5])
cox_surv = estimator.predict_cumulative_hazard_function((X_test[:5]))
for i, surv_func in enumerate(rsf_surv):
    plt.step(time_points, surv_func(time_points), where="post",
             label=" rsf Sample %d" % (i + 1))
for i , surv_func in enumerate(cox_surv):
    plt.step(time_points, surv_func(time_points), where = "post",
             label = "cox sample %d" % (i+1))

plt.ylabel("Cumulative hazard")
plt.xlabel("Time in days")
plt.legend(loc="best")
plt.grid (True)
plt.show()

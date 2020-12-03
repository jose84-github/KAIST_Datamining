import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

#1.데이터 이해
df = pd.read_csv('data_pepTestCustomers.csv')
print('데이터 형태')
df.info()
df.head(10)
print('숫자형 데이터 특징')
df.describe()
df.hist(bins=30, figsize=(20,15))
print('변수 상관관계')
df.corr()
df.corr().pep.sort_values(ascending=False)
plt.matshow(df.corr())
print('이상치 탐색')
df.loc[:,['age', 'income']].plot.box(subplots=True, layout=(2,1),figsize=(10,10))

#2.데이터 준비
## 예측을 위한 기존 X 변수들을 활용한 새로운 변수 생성
#1) 가족수(부양가족수))
df['dependents1'] = df['married']+df['children']+1
df['dependents2'] = df['married']+df['children']

#2) 실질소득금액 (부양가족수로 나눴을 경우 pep와의 Corr값이 떨어져서 그냥 Children으로 나눔)
df['realincome1'] = np.where(df['children']==0, df['income'], df['income']/df['children'])
df['realincome2'] = df['income']/df['dependents1']
df['realincome3'] = df['married']+df['children']*0.5+1
df['realincome4'] = np.where(df['married']==0, df['income'], df['income']/2)
df['realincome5'] = np.where(df['dependents2']==0, df['income'], df['income']/df['dependents2'] )

#3) 기대 수익 (각 개인의 나이별 단위 소득금액 x 정년 퇴직나이65세 = 최대 얼마까지 벌수 있는지를 나타냄)
df['exp_income'] = (df['income']/df['age'])*65
df['exp_income2'] = df['income']*(65-df['age'])

#4) 적금은 없지만 모기지만 있는 개인들의 경우 (하우스푸어) Pep와의 corr값이 높음
df['housepoor'] = np.where((df['save_act']==0) & (df['mortgage']==1) , 1, 0)

#5) 기존 고객(거래가 있는) 여부에 따라 각 거래에 가중치를 부여
df['transaction'] = (df['save_act'])*0.7 + (df['current_act'])*0.1 + (df['mortgage'])*0.2

#6) 범주형 데이터인 지역(region) 정보를 인코딩하여 추가하고 region 삭제
region_enc=pd.get_dummies(df.region)
region_enc.columns=['region_0', 'region_1', 'region_2', 'region_3']
df=pd.concat([df,region_enc], axis=1)
df.drop('region', axis = 1, inplace=True)

#7) 자녀 유무
df['children_YN'] = np.where(df['children']==0, 0, 1)
#df.drop(['children','car','save_act','current_act','mortgage'], axis = 1, inplace=True)
df.head(10)

## train / test data 분리
dfx = df.drop(['id', 'pep'], axis = 1)
dfy = df['pep']

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.25, random_state = 0)
x_train.head(10)
x_train.shape
x_test.shape

## x_train 데이터를 기준으로 x_train과 x_test 데이터를 nomalizing
scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#feature importance
n_feature = dfx.shape[1]
index = np.arange(n_feature)
plt.barh(index, rf.feature_importances_, align='center')
plt.yticks(index, dfx.columns)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()

#3.모델링 + 평가
# RandomForest 실행
rf = RandomForestClassifier(n_estimators = 500, max_depth = 10)
rf.fit(x_train_scaled, y_train)
predicted = rf.predict(x_test_scaled)
print(predicted)

# GradientBoostingClassifier (앙상블 - 그라디언트 부스트)
gb = GradientBoostingClassifier(learning_rate=0.05, n_estimators = 1000)
gb.fit(x_train_scaled, y_train)

# Voting (앙상블 - Voting)
voting_model = VotingClassifier(estimators=[('GradientBoostingClassifier', gb), ('AdaBoostClassifier', ab), ('XGBClassifier',xgb), ('RandomForestClassifier', rf)], voting='soft')
voting_model.fit(x_train_scaled, y_train)

print('Random Forest score is %s'%(rf.score(x_test_scaled, y_test)))
print('Gradientboost score is %s'%(gb.score(x_test_scaled, y_test)))
print('Voting score is %s'%(voting_model.score(x_test_scaled, y_test)))
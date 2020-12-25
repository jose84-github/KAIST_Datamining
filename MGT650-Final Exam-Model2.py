### 데이터마이닝 실습### 연금상품 가입 예측 모델링 - 랜덤포레스트/Catboost/Voting

### 1. 필요 모듈 임포트

#기본으로 사용할 모듈
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#전 처리용 모듈
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

#적용시킬 다양한 모델
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#옵션 최적화를 위한 모듈
from sklearn.model_selection import GridSearchCV, KFold

#현재 폴더 이동후 데이터 로딩
os.chdir(r'C:\Users\JAMHouse\Documents\MBA_Datamining')
df = pd.read_csv('data_pepTestCustomers.csv')
df.head(3)

### 2. 데이터 이해

## 가. 데이터 형태
df.info()
df.head(10)

## 나. 숫자형 데이터 특징
df.describe()
df.hist(bins=30, figsize=(20,15))

## 다. 변수간 상관관계
df.corr()
df.corr().pep.sort_values(ascending=False)
plt.matshow(df.corr())

## 라. 이상치 탐색
df.loc[:,['age', 'income']].plot.box(subplots=True, layout=(2,1),figsize=(10,10))

## 마. 자녀수에 따른 연금보험 가입률
sns.barplot(y='pep', x='children', data=df)

### 3. 데이터 준비

## 가. 예측을 위한 기존 X 변수들을 활용한 새로운 변수 생성

# 1) 가족수(부양가족수)
df['dependents1'] = df['married']+df['children']+1
df['dependents2'] = df['married']+df['children']

# 2) 실질소득금액 
df['realincome1'] = np.where(df['children']==0, df['income'], df['income']/df['children'])
#df['realincome2'] = df['income']/df['dependents1']
#df['realincome3'] = df['married']+df['children']*0.5+1
#df['realincome4'] = np.where(df['married']==0, df['income'], df['income']/2)
#df['realincome5'] = np.where(df['dependents2']==0, df['income'], df['income']/df['dependents2'] )

# 3) 기대 수익
#df['exp_income'] = (df['income']/df['age'])*65
df['exp_income2'] = df['income']*(65-df['age'])

# 4) 적금은 없지만 모기지만 있는 개인들의 경우 (하우스푸어) Pep와의 corr값이 높음
df['housepoor'] = np.where((df['save_act']==0) & (df['mortgage']==1) , 1, 0)

# 5) 기존 고객(거래가 있는) 여부에 따라 각 거래에 가중치를 부여
df['transaction'] = (df['save_act'])*0.7 + (df['current_act'])*0.1 + (df['mortgage'])*0.2

# 6) 지역(region): 범주형 데이터인 지역 정보를 인코딩하여 추가하고 region 삭제
region_enc=pd.get_dummies(df.region)
region_enc.columns=['region_0', 'region_1', 'region_2', 'region_3']
df=pd.concat([df,region_enc], axis=1)
df.drop('region', axis = 1, inplace=True)

# 7) 자녀(children): 자녀 유무, 자녀 정보를 범주형 데이터로 인코딩 후 children 삭제
#df['children_YN'] = np.where(df['children']==0, 0, 1)
#df.drop('children', axis = 1, inplace=True)
children_enc=pd.get_dummies(df.children)
children_enc.columns=['children_0', 'children_1', 'children_2', 'children_3']
df=pd.concat([df,children_enc], axis=1)
df.drop('children', axis = 1, inplace=True)

## 나. X 변수와 Y 변수 분리
dfx = df.drop(['id', 'pep'], axis = 1)
dfy = df['pep']

## 다. train / test data 분리
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.3, random_state = 0)

## 라. x_train 데이터를 기준으로 x_train과 x_test 데이터를 nomalizing
# MinMax 방식이 Z-score 방식보다 우수하여 이를 선택
# 전체 변수에 대해서 normalizing 실시
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

### 4. 모델링 및 예측 정확성 산출
## 가. RandomForest
# 모델 구축 및 평가
rf = RandomForestClassifier(n_estimators = 500, max_depth = 9, random_state=0)
rf.fit(x_train_scaled, y_train)
predicted = rf.predict(x_test_scaled)
#print(predicted)
print('Random Forest: score is %s'%(rf.score(x_test_scaled, y_test)))

# 옵션 최적화
'''
param_grid={'n_estimators' :[500, 1000],
                 'criterion':['gini', 'entropy'],
                 'max_depth':[8, 9, 10, 11, 12],
                 'max_features':['auto', 'sqrt', 'log2'],
                 'class_weight':['balanced', 'balanced_subsample'] }
cv=KFold(n_splits=6, random_state=0)
gcv=GridSearchCV(rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=4)
gcv.fit(x_train_scaled, y_train)
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수
'''

## 나. Categorical Boosting(CatBoost)
# 모델 구축 및 평가
cb = CatBoostClassifier(iterations= 39, random_seed = 0, silent=True)
cb.fit(X = x_train_scaled, y = y_train, silent=True)
print('CatBoost: score is %s'%(cb.score(x_test_scaled, y_test)))

# 옵션 최적화
'''
param_grid={'iterations' :list(range(1,100))}
cv=KFold(n_splits=6, random_state=0)
gcv=GridSearchCV(cb, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=4)
gcv.fit(x_train_scaled, y_train)
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수
'''

## 다. GradientBoostingClassifier
# 모델 구축 및 평가
gb = GradientBoostingClassifier(loss='exponential', n_estimators = 15, criterion='friedman_mse', max_features='auto', random_state=0)
gb.fit(x_train_scaled, y_train)

# 옵션 최적화
'''
param_grid={'n_estimators' :list(range(1,100)),
                 'loss':['exponential', 'deviance'],
                 'criterion' : ['friedman_mse', 'mse', 'mae'],
                 'max_features': ['auto', 'sqrt', 'log2']}
cv=KFold(n_splits=6, random_state=0)
gcv=GridSearchCV(gb, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=4)
gcv.fit(x_train_scaled, y_train)
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수
'''

## 라. Extreme Gradient Boosting(XGB)
# 모델 구축 및 평가
xgb = XGBClassifier(booster='gbtree', colsample_bylevel=0.9, colsample_bytree=0.8, gamma=2, max_depth=6, min_child_weight=2, n_estimators=1000, nthread=4, objective='binary:logistic', random_state=0)
xgb.fit(x_train_scaled, y_train)

# 옵션 최적화
'''
param_grid={'booster' :['gbtree'],
                 'silent':[True],
                 'max_depth':[6,10,11],
                 'min_child_weight':[1,2,6],
                 'gamma':[1,2,3],
                 'nthread':[4,5],
                 'colsample_bytree':[0.7, 0.8],
                 'colsample_bylevel':[0.8, 0.9],
                 'n_estimators':[500],
                 'objective':['binary:logistic'],
                 'random_state':[0]}
cv=KFold(n_splits=6, random_state=1)
gcv=GridSearchCV(xgb, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=4)
gcv.fit(x_train,y_train)
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수
'''
## 마. Multi-Layer Perceptron(MLP)
# 모델 구축 및 평가
mlp =  MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(8,8), learning_rate='adaptive', max_iter = 1000, random_state=0)
mlp.fit(x_train_scaled, y_train)

## 바. Voting(RF, GB, XGB, CatBoost, MLP)
# 모델 구축 및 평가
voting_model = VotingClassifier(estimators=[('MLPClassifier', mlp), ('XGBClassifier', xgb), ('GradientBoostingClassifier', gb), ('CatBoostClassifier',cb), ('RandomForestClassifier', rf)], voting='hard')
voting_model.fit(x_train_scaled, y_train)
print('Voting: score is %s'%(voting_model.score(x_test_scaled, y_test)))

### 5. feature importance
n_feature = dfx.shape[1]
index = np.arange(n_feature)

## 가. Random Forest Feature Importances
ftr_importances = pd.Series(rf.feature_importances_, index = x_train.columns)
plt.figure(figsize=(8,6))
plt.title('Random Forest Feature Importances')
sns.barplot(x=ftr_importances.sort_values(ascending=False), y=ftr_importances.sort_values(ascending=False).index)
plt.show()

## 나. CatBoost Feature Importances
ftr_importances = pd.Series(cb.feature_importances_, index = x_train.columns)
plt.figure(figsize=(8,6))
plt.title('CatBoost Feature Importances')
sns.barplot(x=ftr_importances.sort_values(ascending=False), y=ftr_importances.sort_values(ascending=False).index)
plt.show()
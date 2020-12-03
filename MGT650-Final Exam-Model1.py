import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#고객 데이터 이해
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

#attributes = ["realincome","extra_money","pep"]
#scatter_matrix(df[attributes], figsize=(12,8))

#고객 데이터 전처리
# def feture_scaling(df, scaling_strategy="min-max", column=None):
#     if column == None:
#         column = [column_name for column_name in df.columns]
#     for column_name in column:
#         if scaling_strategy == "min-max":
#             df[column_name] = ( df[column_name] - df[column_name].min() ) /\
#                             (df[column_name].max() - df[column_name].min()) 
#         elif scaling_strategy == "z-score":
#             df[column_name] = ( df[column_name] - \
#                                df[column_name].mean() ) /\
#                             (df[column_name].std() )
#     return df

# bins = [0, 20, 30, 40, 50, 60, 70]
# bins_names = ['10', '20', '30', '40', '50', '60']
# categories = pd.cut(df['age'], bins, labels=bins_names)
# df['age'] = categories
# df['age'] = pd.to_numeric(df['age'])

# data_category = df.loc[:, ['sex','region','married','car','save_act','current_act','mortgage']]
# data_category
# data_dummy = pd.get_dummies(data_category, columns=['sex','region','married','car','save_act','current_act','mortgage'])
# data_dummy
# df = df.drop(columns=['sex','region','married','car','save_act','current_act','mortgage'])
# df = pd.concat([df, data_dummy],axis=1)

# df['dependents'] = df['married_1']+df['children']+1
# df['realincome'] = np.where(df['children']==0, df['income'], df['income']/df['children'])
# df['exp_income'] = (df['income']/df['age'])*65
# df['housepoor'] = np.where((df['save_act_1']==0) & (df['mortgage_1']==1) , 1, 0)
# df['transaction'] = (df['save_act_1'])*0.7 + (df['current_act_1'])*0.1 + (df['mortgage_1'])*0.2

#df['expense_lv'] = (df['car'])*0.1 + (df['save_act'])*0.7 + (df['mortgage'])*0.2
#df['unmar_weight'] = (30-df['age'])*(1-df['married'])
#df['car'] = df['dependents']*df['car']
#df['save_act'] = df['dependents']*df['save_act']
#df['mortgage'] = df['dependents']*df['mortgage']
#df['transaction'] = (df['save_act'])*0.7 + (df['current_act'])*0.1 + (df['mortgage'])*0.2
#df['unmarried_init'] = np.where((df['married']==0) & (df['children']==0) & (df['save_act']==0) , 1, 0)
#df['one_child'] = np.where((df['children']==1), 1, 0)
#df['three_child'] = np.where((df['children']==3), 1, 0)

# feture_scaling(df,column=['age','dependents','realincome','exp_income'])
# df = df.drop(['income','children'],axis=1)
# df.head(10)

#Training - Test Set 만들기
dfx = df.drop(['id','pep'],axis=1)
dfy=df['pep']

x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.3, random_state=0)
x_train.shape
x_test.shape

#DCS Tree 만들기

dcs_tree = DecisionTreeClassifier(max_depth=6, random_state=0)
dcs_tree.fit(x_train, y_train)

predicted = dcs_tree.predict(x_test)
print(predicted)
print('score is %s'%(dcs_tree.score(x_test, y_test)))

# #DCS Tree 시각화

# from sklearn.tree import export_graphviz
# import graphviz

# export_graphviz(dcs_tree, out_file='iris_tree.dot')
# # dot = graphviz.Source(iris)
# # dot.format='png'
# # dot.render(filename='tree.png')

# import graphviz
# from IPython.display import display 

# with open("iris_tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))

#DCS Tree 특성 중요도

print("특성 중요도:\n{}".format(dcs_tree.feature_importances_))
feature_names = dfx[0:9]

def plot_feature_importances_pep(model):
    n_features = dfx.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("특성 중요도", fontname = 'Malgun Gothic')
    plt.ylabel("특성", fontname = 'Malgun Gothic')
    plt.ylim(-1, n_features)

plot_feature_importances_pep(dcs_tree)
import pandas as pd

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

df = pd.read_csv('data_pepTestCustomers.csv')
df.head(3)

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

# feture_scaling(df,column=['age','dependents','realincome','exp_income'])
# df = df.drop(['income','children'],axis=1)

from sklearn.model_selection import train_test_split
dfx = df.drop(['id','pep'], axis=1)
dfy = df['pep']
dfx.head(3)

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, random_state=0)
x_train.shape
x_test.shape

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100, max_depth=6)
rf.fit(x_train, y_train)
predicted = rf.predict(x_test)

print(predicted)
print('score is %s'%(rf.score(x_test, y_test)))

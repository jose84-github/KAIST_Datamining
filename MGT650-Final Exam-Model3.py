import pandas as pd

def feture_scaling(df, scaling_strategy="min-max", column=None):
    if column == None:
        column = [column_name for column_name in df.columns]
    for column_name in column:
        if scaling_strategy == "min-max":
            df[column_name] = ( df[column_name] - df[column_name].min() ) /\
                            (df[column_name].max() - df[column_name].min()) 
        elif scaling_strategy == "z-score":
            df[column_name] = ( df[column_name] - \
                               df[column_name].mean() ) /\
                            (df[column_name].std() )
    return df

df = pd.read_csv('data_pepTestCustomers.csv')
df.head(3)
#feture_scaling(df,column=["age","income","children"])
#df.head(10)

from sklearn.model_selection import train_test_split
dfx = df.drop(['id','pep'], axis=1)
dfy = df['pep']
dfx.head(3)

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.3, random_state=0)
x_train.shape
x_test.shape

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100, max_depth=6)
rf.fit(x_train, y_train)
predicted = rf.predict(x_test)

print(predicted)
print('score is %s'%(rf.score(x_test, y_test)))

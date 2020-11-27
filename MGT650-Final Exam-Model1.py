import pandas as pd

#DCS Tree 데이터 전처리 함수
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
#feture_scaling(df,column=["region"])
feture_scaling(df,column=["age","region","income","children"])
df.head(10)

#Training - Test Set 만들기
from sklearn.model_selection import train_test_split
dfx = df.drop(['id','pep'],axis=1)
dfx.head(3)
dfy=df['pep']

x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size=0.3, random_state=0)
x_train.shape
x_test.shape

#DCS Tree 만들기

from sklearn.tree import DecisionTreeClassifier

dcs_tree = DecisionTreeClassifier(max_depth=6, random_state=0)
dcs_tree.fit(x_train, y_train)

predicted = dcs_tree.predict(x_test)
print(predicted)
print('score is %s'%(dcs_tree.score(x_test, y_test)))

#DCS Tree 시각화

from sklearn.tree import export_graphviz
import graphviz

export_graphviz(dcs_tree, out_file='iris_tree.dot')
"""dot = graphviz.Source(iris)
dot.format='png'
dot.render(filename='tree.png') """

import graphviz
from IPython.display import display 

with open("iris_tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

#DCS Tree 특성 중요도
import matplotlib.pyplot as plt
import numpy as np

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
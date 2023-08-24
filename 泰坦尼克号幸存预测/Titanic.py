# 引用pandas，sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# %%

# %%
# 读取train文件中的数据
data = pd.read_csv('data/train.csv')
df = data.copy()
df.sample(10)
# %%

# %%
# 去除无用特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
# %%

# %%
# 替换/删除空值，这里是删除
df.dropna(inplace=True)
# %%

# %%
# 把categorical数据通过one-hot变成数值型数据
# 对性别编码
sexdict = {'male':1, 'female':0}
df.Sex = df.Sex.map(sexdict)
# 对embarked编码
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( df['Embarked'] , prefix='Embarked' )
df = pd.concat([df,embarkedDf],axis=1)
df.drop('Embarked',axis=1,inplace=True)
# 对pclass编码
pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies( df['Pclass'] , prefix='Pclass' )
df = pd.concat([df,pclassDf],axis=1)
df.drop('Pclass',axis=1,inplace=True)
df = pd.get_dummies(df)
# 对人数进行计算随行人员加1
df['family']=df.SibSp+df.Parch+1
# df.head(1)
# 划分特征和标签
X = df.drop('Survived', axis=1)
y = df['Survived']
# %%

# %%
# 划分训练集和测试集，取train中的百分之八十用于建模，其余用来预测
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# 创建决策树模型
model = DecisionTreeClassifier()
# 拟合模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
# %%



# 项目读取train文件中的数据
# 去除了'PassengerId', 'Name', 'Ticket', 'Cabin'等无用特征，对非数值型数据采用one-hot编码
# 最终使用决策树模型进行拟合，计算出准确率
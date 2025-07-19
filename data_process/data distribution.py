import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载训练数据
train_path = "C:\\Users\\HUAWEI\\Desktop\\cnews.train.txt"
train_df = pd.read_csv(train_path, sep='\t', names=['label', 'content'], encoding='utf-8')

# 查看前5行
print("训练数据样例：")
print(train_df.head())

# 检查数据规模
print("\n数据规模：", train_df.shape)
print("各列数据类型：\n", train_df.dtypes)

# 统计类别分布
class_dist = train_df['label'].value_counts()

# 绘制类别分布图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
sns.barplot(x=class_dist.index, y=class_dist.values, palette="viridis")
plt.title("新闻类别分布")
plt.xlabel("")
plt.ylabel("样本数量")
plt.xticks(rotation=45)
plt.show()

# 输出类别占比
print("\n类别占比：")
print(class_dist / len(train_df))

# 计算每条新闻的字符长度
train_df['char_length'] = train_df['content'].apply(len)

# 按类别分组统计
print("\n各类别平均字符长度：")
print(train_df.groupby('label')['char_length'].mean())


# 绘制长度分布箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(x='label', y='char_length', data=train_df, showfliers=False)
plt.title("各类别新闻长度分布")
plt.xticks(rotation=45)
plt.show()

# 检查缺失值
print("\n缺失值统计：")
print(train_df.isnull().sum())



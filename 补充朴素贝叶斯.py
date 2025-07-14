import pandas as pd
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)


# 数据加载
def load_data():
    train_df = pd.read_csv("C:\\Users\\HUAWEI\\Desktop\\cnews.train.txt",
                           sep='\t', names=['label', 'content'])
    test_df = pd.read_csv("C:\\Users\\HUAWEI\\Desktop\\cnews.test.txt",
                          sep='\t', names=['label', 'content'])
    print("训练集样本数:", len(train_df))
    print("测试集样本数:", len(test_df))
    return train_df, test_df


# 中文分词处理
def chinese_word_cut(text_series):
    print("\n开始中文分词...")
    jieba.initialize()
    return text_series.apply(lambda x: ' '.join(jieba.cut(x)))


# 加载停用词
def load_stopwords(stopwords_path):
    with open(stopwords_path, encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]
    print("加载停用词数量:", len(stopwords))
    return stopwords


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.title('混淆矩阵 - 补充朴素贝叶斯', fontsize=16)
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 构建模型管道
def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=stopwords,
            max_features=5000,
            lowercase=False,
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            sublinear_tf=True,
            max_df=0.8,
            min_df=5
        )),
        ('selector', SelectKBest(chi2, k=3000)),  # 选择最好的1000个特征
        ('model', ComplementNB(
            alpha=1.0,  # 平滑参数
            norm=True,  # 标准化权重
            class_prior=None  # 自动计算类先验
        ))
    ])
    return pipeline


# 主函数
def main():
    # 1. 加载数据
    train_df, test_df = load_data()

    # 2. 加载停用词
    global stopwords
    stopwords = load_stopwords('C:\\Users\\HUAWEI\\Desktop\\cnews.vocab.txt')

    # 3. 中文分词
    X_train = chinese_word_cut(train_df['content'])
    y_train = train_df['label']
    X_test = chinese_word_cut(test_df['content'])
    y_test = test_df['label']
    classes = np.unique(y_train)

    # 4. 构建并训练模型
    print("\n构建补充朴素贝叶斯模型...")
    model = build_model()
    model.fit(X_train, y_train)

    # 5. 评估模型
    y_pred = model.predict(X_test)

    print("\n模型评估结果:")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, classes)


if __name__ == "__main__":
    main()

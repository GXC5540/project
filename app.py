from flask import Flask, request, render_template, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
import jieba
import pandas as pd
import pickle
import os

app = Flask(__name__)

# 模型保存路径
MODEL_DIR = "D:\\pycharm2\\pythonProject2\\BYS"
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier_model.pkl")


def chinese_word_cut(text_series):
    jieba.initialize()
    return text_series.apply(lambda x: ' '.join(jieba.cut(x)))


def initialize_model():
    # 如果模型已存在，则直接加载
    if os.path.exists(MODEL_PATH):
        print("加载已保存的模型...")
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

    # 确保目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 加载停用词
    with open("C:\\Users\\HUAWEI\\Desktop\\cnews.vocab.txt", encoding='utf-8') as f:
        stopwords = [line.strip() for line in f if line.strip()]

    # 构建模型管道
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words=stopwords,
            max_features=20000,
            lowercase=False,
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_df=0.7,
            min_df=10
        )),
        ('selector', SelectKBest(chi2, k=12000)),
        ('model', ComplementNB(
            alpha=1.0,
            norm=True,
            class_prior=None
        ))
    ])

    # 加载训练数据并训练模型
    print("训练新模型...")
    train_df = pd.read_csv("C:\\Users\\HUAWEI\\Desktop\\cnews.train.txt", sep='\t', names=['label', 'content'])
    X_train = chinese_word_cut(train_df['content'])
    y_train = train_df['label']

    model.fit(X_train, y_train)

    # 保存训练好的模型
    print(f"保存模型到 {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model


# 加载模型
model = initialize_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取用户输入的新闻内容
        news_content = request.form['news_content']

        # 分词处理
        segmented_content = ' '.join(jieba.cut(news_content))

        # 预测分类
        prediction = model.predict([segmented_content])[0]

        # 返回预测结果
        return jsonify({
            'status': 'success',
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=5000)

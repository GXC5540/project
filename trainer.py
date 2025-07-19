from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif,f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_CONFIG, PLOT_CONFIG


class Trainer:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = [PLOT_CONFIG['font']]
        plt.rcParams['axes.unicode_minus'] = False

    '''def vectorize_text(self, X_train, X_test, stopwords):

        tfidf = TfidfVectorizer(
            stop_words=stopwords,
            max_features=MODEL_CONFIG['max_features'],
            ngram_range=MODEL_CONFIG['ngram_range'],
            max_df=MODEL_CONFIG['max_df'],
            min_df=MODEL_CONFIG['min_df'],
            lowercase=False,
            sublinear_tf=True
        )
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)
        return X_train_vec, X_test_vec, tfidf'''
    def vectorize_text(self, X_train, X_test, stopwords):

        bow = CountVectorizer(
            stop_words=stopwords,
            max_features=MODEL_CONFIG['max_features'],
            ngram_range=MODEL_CONFIG['ngram_range'],
            max_df=MODEL_CONFIG['max_df'],
            min_df=MODEL_CONFIG['min_df'],
            lowercase=False,
        )
        X_train_vec = bow.fit_transform(X_train)
        X_test_vec = bow.transform(X_test)
        return X_train_vec, X_test_vec, bow

    def select_features(self, X_train, y_train, X_test):
        """特征选择"""
        selector = SelectKBest(chi2, k=MODEL_CONFIG['k_best'])
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        return X_train_sel, X_test_sel, selector

    def evaluate(self, y_true, y_pred, classes, model_name):
        """模型评估"""
        print(f"\n{model_name} 评估结果:")
        print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
        print("\n分类报告:\n", classification_report(y_true, y_pred))

        # 绘制混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=PLOT_CONFIG['figsize'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=PLOT_CONFIG['cmap'],
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'{model_name} - 混淆矩阵', fontsize=16)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
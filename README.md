朴素贝叶斯新闻分类。本项目借助朴素贝叶斯和文本向量化实现了一个新闻分类系统，支持以下新闻类别：体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技和财经。
实现了网页制作。后端：app.py,基于Flask搭建服务，通过TF-IDF提取文本特征，用SelectKBest筛选特征，结合ComplementNB模型训练分类器，支持加载/训练模型，提供预测接口接收文本并返回分类结果。
前端：index.html,用HTML+Tailwind CSS构建页面，支持文本输入、分类请求，通过JS处理交互，展示预测结果、置信度及历史记录，还实现深色模式切换等功能。
config里面是模型的相关参数；multinomialNB_model和ComplementNB_model是模型构建；data_loader是数据处理，分词和去除停用词；trainer是训练模型，包含TF-IDF和bow两种向量化方法，特征选择，绘制混淆矩阵。
训练好的模型参数保存在BYS里面
数据请联系作者1779963176@stu.xjtu.edu.cn获取

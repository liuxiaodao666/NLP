import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 读取txt文件
with open("D:\code\data_path\cnews\cnews.train.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

data = []
for line in lines:
    # 假设每行文本与标签之间使用制表符（\t）分隔
    label, text = line.strip().split("\t")
    data.append({"text": text, "label": label})

# 将数据转换为DataFrame
data = pd.DataFrame(data)

# 划分特征和标签
X = data['text']
y = data['label']

# 文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练分类器
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 预测
y_pred = classifier.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

import pandas as pd  # 数据处理
from sklearn.cluster import KMeans  # 机器学习：聚类
from sklearn import metrics  # 机器学习：聚类
import matplotlib.pyplot as plt  # 作图
import matplotlib  # 指定默认字体
import jieba  # 中文分词工具
import re  # 正则表达式
from collections import Counter  # 统计数量
from pyecharts import *  # 可视化渲染图片

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度
pd.set_option('max_colwidth', 100)
# 设置1000列时才换行
pd.set_option('display.width', 1000)
# 解决pd.plotting坐标轴无法正常显示中文问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

df = pd.read_excel('优雅-哔哩哔哩_Bilibili_cleaned.xlsx')

# 运用describe()总体分析数值型数据列
print(df.describe())

# 以“优雅”为主题的视频发布时间分析
df = df[pd.notnull(df['发布时间'])]
df1 = df.copy()
df1.sort_values(by='发布时间', axis=0, ascending=False, inplace=True)
print(df1.head(10))

# 统计各年份发布的视频数量
df1['发布时间'] = df1['发布时间'].str[0:4]
print(pd.DataFrame(df1['发布时间'].value_counts()))
print('\n')
# 视频标题、标签词云分析
# 视频标签词云分析
strings = ''
for item in df['标签']:
    strings += item
    strings += " "
# 先统计一下标签词频
words = strings.split()
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
items = list(counts.items())
items.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    word, count = items[i]
    print("{0:<10}{1:>5}".format(word, count))

keys, values = Counter(words).keys(), Counter(words).values()
wordcloud = WordCloud('视频标签词云', width=900, height=620)
wordcloud.add("视频标签词汇词频", list(keys), list(values), word_size_range=[10, 100])
wordcloud.render(path="./视频标签词云.html")

# 视频标题词云分析
strings1 = ''
for item in df['视频标题']:
    strings1 += item
    strings1 += " "
strings1 = ''.join(re.findall(r'[\u4e00-\u9fa5]', strings1))  # 提取全部的中文词汇
words1 = jieba.lcut(strings1)  # 将strings用jieba库切分成words list

keys, values = Counter(words1).keys(), Counter(words1).values()
wordcloud = WordCloud('视频标题词云', width=900, height=620)
wordcloud.add("视频标题词汇词频", list(keys), list(values), word_size_range=[10, 100])
wordcloud.render(path="./视频标题词云.html")

# 机器学习：无监督学习-聚类
analysis_data = df1[['视频标题', '总播放数', '总弹幕数', '点赞数', '投币数', '收藏数']].copy()

# 通过绘制特征散点图矩阵，观察每两种特征的区分度
pd.plotting.scatter_matrix(analysis_data, diagonal='hist')
plt.show()

# 定义簇的个数为2，取后5列特征值，训练聚类模型
X = analysis_data.iloc[:, 1:6].values.astype(int)  # 准备数据
kmeans = KMeans(n_clusters=2)  # 模型初始化
kmeans.fit(X)  # 训练模型

# 使用样本簇编号作为类型标签，绘制特征对的散点图矩阵用不同颜色标识不同的簇
pd.plotting.scatter_matrix(analysis_data, c=kmeans.labels_, diagonal='hist')
plt.show()

# 使用轮廓系数（Silhouette Coefficient）来度量聚类的质量
print(metrics.silhouette_score(X, kmeans.labels_, metric='euclidean'))

# 尝试多个k值聚类，比较轮廓系数
clusters = [2, 3, 4, 5, 6, 7, 8]
sc_scores = []
# 计算各个簇模型的轮廓系数
for i in clusters:
    kmeans = KMeans(n_clusters=i).fit(X)
    sc = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
    sc_scores.append(sc)
# 绘制曲线图反应轮廓系数与簇数的关系
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()

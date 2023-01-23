import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度
pd.set_option('max_colwidth', 100)
# 设置1000列时才换行
pd.set_option('display.width', 1000)


def str2value(valueStr):  # 数据类型转换，字符串→值
    valueStr = str(valueStr)
    idxOfWan = valueStr.find('万')
    iftoubi = valueStr.find('投币')
    ifshoucang = valueStr.find('收藏')
    iffenxiang = valueStr.find('分享')
    if iffenxiang != -1:
        return 0
    elif ifshoucang != -1:
        return 0
    elif iftoubi != -1:
        return 0
    elif idxOfWan != -1:
        return int(float(valueStr[:idxOfWan]) * 1e4)
    elif idxOfWan == -1:
        return float(valueStr)


def str2sec(x):
    """
    字符串分秒转换成秒
    """
    if not pd.notnull(x):
        return x
    else:
        x = str(x)
        m, s = x.strip().split(':')  # .split()函数将其通过':'分隔开，.strip()函数用来除去空格
        return int(m) * 60 + int(s)  # int()函数转换成整数运算


df = pd.read_excel('优雅-哔哩哔哩_Bilibili.xlsx')
# print(df)
df.rename(columns={'字段': '转发数'}, inplace=True)
df = df.drop(['全站排行榜情况'], axis=1)

df['总播放数'] = df['总播放数'].apply(str2value)
df['总弹幕数'] = df['总弹幕数'].apply(str2value)
df['点赞数'] = df['点赞数'].apply(str2value)
df['投币数'] = df['投币数'].apply(str2value)
df['收藏数'] = df['收藏数'].apply(str2value)
df['转发数'] = df['转发数'].apply(str2value)

df['视频时长'] = df['视频时长'].apply(str2sec)
df.drop_duplicates(subset=['页面网址'], keep='first', inplace=True)
df.to_excel("优雅-哔哩哔哩_Bilibili_cleaned.xlsx", index=False)


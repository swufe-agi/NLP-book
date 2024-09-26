import gensim.downloader as api

# 加载使用维基百科数据训练的模型，66MB
model = api.load("glove-wiki-gigaword-50")

print(model.most_similar("king"))
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords

# 步骤 1：读取文件
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# 步骤 2：预处理文本
def preprocess(text):
    return [simple_preprocess(remove_stopwords(line)) for line in text]

# 步骤 3：训练模型
def train_word2vec(data):
    model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1)
    return model

file_path = 'alice.txt'
raw_text = read_file(file_path)
processed_data = preprocess(raw_text)
model = train_word2vec(processed_data)

print(model.wv.similarity('hole', 'rabbit'))
print(model.wv.similarity('wonderland', 'rabbit'))
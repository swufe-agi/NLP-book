from collections import defaultdict
import re


def preprocess(text):
    # 简单的预处理：分割成句子，转小写，去除标点
    sentences = re.split(r'[.!?]+', text.lower().strip())
    return [['<START>'] + sentence.split() + ['<END>'] for sentence in sentences if len(sentence) > 3]


def build_bigram_model(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    sentences = preprocess(text)

    # 统计bigram和unigram频率
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            bigram_counts[bigram] += 1
            unigram_counts[sentence[i]] += 1

    # 计算条件概率
    bigram_probs = {}
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[(w1, w2)] = count / unigram_counts[w1]

    return bigram_probs


def sentence_probability(sentence, bigram_probs):
    words = ['<START>'] + sentence.lower().split() + ['<END>']
    probability = 1.0
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        if bigram in bigram_probs:
            probability *= bigram_probs[bigram]
        else:
            # 处理未见过的bigram，可以使用平滑技术
            probability *= 1e-10  # 简单起见，这里使用一个很小的概率
    return probability


if __name__ == '__main__':
    file_path = 'cat.txt'
    bigram_probs = build_bigram_model(file_path)
    test_sentences = ["The cat is a pet", "The cat is a king"]
    for s in test_sentences:
        prob = sentence_probability(s, bigram_probs)
        print(f"The probability of '{s}' is {prob}")

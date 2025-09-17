import gensim.downloader as api

wv = api.load("word2vec-google-news-300")

for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")


vec_king = wv["king"]
print("vector for 'king':\n", vec_king)

pairs = [
    ("car", "minivan"),  # a minivan is a kind of car
    ("car", "bicycle"),  # still a wheeled vehicle
    ("car", "airplane"),  # ok, no wheels, but still a vehicle
    ("car", "cereal"),  # ... and so on
    ("car", "communism"),
]
for w1, w2 in pairs:
    print("%r\t%r\t%.2f" % (w1, w2, wv.similarity(w1, w2)))


print(wv.most_similar(positive=["car", "minivan"], topn=5))

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
import faiss
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    '私はプログラマーです。',
    '私はWebデベロッパーです。',
    '私はシステムエンジニアです。',
    '私はエンジニアです。']


embeddings = model.encode(sentences)

for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("Embedding Dimension:", len(embedding))
    print("----------------\n")


paraphrases = util.paraphrase_mining(model, sentences)

for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
print("----------------\n")


tsne = TSNE(n_components=2, perplexity=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
for i, sentence in enumerate(sentences):
    plt.annotate(sentence, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))
plt.show()


# https://faiss.ai/
index = faiss.IndexFlatIP(384)
print(index.is_trained)
index.add(embeddings)
print(index.ntotal)

D, I = index.search(model.encode(['Software Design']), 1)
print(D)
print(I)
print(I[0][0])
print(sentences[I[0][0]])

import gensim
from gensim.models import Word2Vec
import spacy

# 学習済みモデルとしては以下などがある
# http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
PRETRINED_MODEL_PATH = 'path/to/pretrained_model.bin'

CUSTOM_DATASET_PATH = 'path/to/custom_dataset.txt'

FINE_TUNED_MODEL_PATH = 'path/to/fine_tuned_model.bin'

# 学習済みのword2vecモデルを読み込み
model = gensim.models.KeyedVectors.load_word2vec_format(PRETRINED_MODEL_PATH, binary=True)

# 固有名詞のデータセットを読み込み、GiNZAを使用して分かち書きする
nlp = spacy.load('ja_ginza')
with open(CUSTOM_DATASET_PATH, encoding='utf-8') as f:
    sentences = [doc.text for doc in nlp.pipe(f, batch_size=1000, n_process=4)]

# 固有名詞の単語を新しいレイヤーに追加し、Fine-tuning
for sentence in sentences:
    doc = nlp(sentence)
    for token in doc:
        # TODO このコードは追加のデータセットに既知の固有名詞入っている前提でトークン分割してエンティティタイプを絞って追加している
        # 既に固有名詞のみで構成される場合はトークン分割など不要であるのでそのまま投入すれば良さそう
        # その場合、既に含まれる固有名詞は除外する
        # ex. if word in model.wv.vocab and model.wv.vocab[word].count > 1:

        if token.ent_type_ == "LOC" or token.ent_type_ == "ORG":
            model.wv.vocab[token.text] = gensim.models.word2vec.Vocab(count=1, index=len(model.wv.vocab))
    # TODO 毎度毎度学習するのは非効率なので工夫の余地がある
    model.train([doc], total_examples=1, epochs=1)

# Fine-tuning後のモデルを保存
model.save(FINE_TUNED_MODEL_PATH)

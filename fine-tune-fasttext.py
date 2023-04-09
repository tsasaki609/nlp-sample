import fasttext

# 学習済みモデルとしては以下などがある
# https://fasttext.cc/docs/en/crawl-vectors.html
PRETRINED_MODEL_PATH = 'path/to/pretrained_model.bin'

CUSTOM_DATASET_PATH = 'path/to/custom_dataset.txt'

FINE_TUNED_MODEL_PATH = 'path/to/fine_tuned_model.bin'

# 学習済みのFastTextモデルを読み込む
pretrained_model = fasttext.load_model(PRETRINED_MODEL_PATH)

# 追加学習用のFastTextモデルを作成する
model = fasttext.train_unsupervised(CUSTOM_DATASET_PATH, model='skipgram')

# 学習済みのFastTextモデルの単語ベクトルを、追加学習用のFastTextモデルにコピーする
for word in pretrained_model.words:
    if word in model.words:
        model[word] = pretrained_model[word]

# 追加学習用のFastTextモデルで、追加学習させるデータを使って学習を行う
model.train_unsupervised(CUSTOM_DATASET_PATH, model='skipgram', epochs=10)

# 追加学習が完了したFastTextモデルを保存する
model.save_model(FINE_TUNED_MODEL_PATH)

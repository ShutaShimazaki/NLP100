### 51. 特徴量抽出　-データの特徴を数値で表す ### 
#学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
#なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
#記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

## ここで必要に応じてテキストを前処理すると精度上がるかも？　##

import sys
sys.path.append("chap6\\obtain_format_50")
from obtain_format_50 import df_train, df_valid, df_test

# TF-IDF：単語の頻度に応じて文章をベクトル化　https://atmarkit.itmedia.co.jp/ait/articles/2112/23/news028.html
    #以下コード：https://kagglenote.com/kaggle/text-vectorize-by-tfidf/
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features= 1000 
    )
    #NLP100解答　https://kakedashi-engineer.appspot.com/2020/05/09/nlp100-ch6/
    #X_train: モデルの説明変数X
X_train = vectorizer.fit_transform(df_train['TITLE'])
X_valid = vectorizer.fit_transform(df_valid['TITLE'])
X_test = vectorizer.fit_transform(df_test['TITLE'])

import numpy as np
np.savetxt('chap6\\csv\\X_train.txt', X_train.toarray(), fmt='%d')
np.savetxt('chap6\\csv\\X_valid.txt', X_valid.toarray(), fmt='%d')
np.savetxt('chap6\\csv\\X_test.txt', X_test.toarray(), fmt='%d')

print(X_train.toarray())
# print(vectorizer.get_feature_names_out())
# print(X_valid.shape)

### 52 学習 ###
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
    # ロジスティック回帰分析 https://gmo-research.jp/research-column/logistic-regression-analysis

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression(max_iter=10000) #max_iterで最大反復回数を指定
lg_model.fit(X_train, df_train['CATEGORY']) #model.fit(データ, 正解ラベル)

### 53 予測 ###
# 52で学習したロジスティック回帰モデルを用い，
# 与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．

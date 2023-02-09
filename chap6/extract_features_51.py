### 51. 特徴量抽出　-データの特徴を数値で表す ### 
#学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
#なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
#記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

# # TF-IDFで単語の頻度を調べる前に処理
# # 1. 記号を省く
# # 2. 全て小文字へ
# # 3. 数字を０へ変換
# import string
# import re

# def preprocessing(text):
#   table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
#   text = text.translate(table)  #1 記号をスペースに置換
#   text = text.lower()  #2 小文字化
#   text = re.sub('[0-9]+', '0', text)  #3 数字列を0に置換

#   return text

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
features_train = vectorizer.fit_transform(df_train['TITLE'])
features_valid = vectorizer.fit_transform(df_valid['TITLE'])
features_test = vectorizer.fit_transform(df_test['TITLE'])

import numpy as np
np.savetxt('chap6\\csv\\feature_train.txt', features_train.toarray(), fmt='%d')
np.savetxt('chap6\\csv\\feature_valid.txt', features_valid.toarray(), fmt='%d')
np.savetxt('chap6\\csv\\feature_test.txt', features_test.toarray(), fmt='%d')

print(features_train.toarray())
# print(vectorizer.get_feature_names_out())
# print(features_valid.shape)
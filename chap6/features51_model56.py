##### 特徴を数値化、モデルの学習 #####

### 51. 特徴量抽出　-データの特徴を数値で表す ### 
#学習データ，検証データ，評価データから特徴量を抽出し，それぞれtrain.feature.txt，valid.feature.txt，test.feature.txtというファイル名で保存せよ． 
#なお，カテゴリ分類に有用そうな特徴量は各自で自由に設計せよ．
#記事の見出しを単語列に変換したものが最低限のベースラインとなるであろう．

## ここで必要に応じてテキストを前処理すると精度上がるかも？　##

import sys
sys.path.append("chap6\\obtain_format_50")
sys.path.append('C:\\Users\\user.DESKTOP-JAJ50S0\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\Scripts')

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

# print(vectorizer.get_feature_names_out())
# print(X_valid.shape)

### 52 学習 ###
# 51で構築した学習データを用いて，ロジスティック回帰モデルを学習せよ．
    # ロジスティック回帰分析の概要 https://gmo-research.jp/research-column/logistic-regression-analysis

from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression(max_iter=10000) #max_iterで最大反復回数を指定
lg_model.fit(X_train, df_train['CATEGORY']) #model.fit(データ, 正解ラベル)　fitメソッド：ロジスティック回帰モデルの重みを学習

### 53 予測 ###
# 52で学習したロジスティック回帰モデルを用い，
# 与えられた記事見出しからカテゴリとその予測確率を計算するプログラムを実装せよ．
#https://qiita.com/0NE_shoT_/items/c42d8093e2fed9bf1b7a (ロジスティック関数の実装と評価)

# print(X_train) ->
    #   (0, 306)     0.4076222664213075
    #   (0, 731)      0.43591619896792233
    #   (0, 806)      0.41399909575175514
    #   (0, 68)       0.22995365292038036
    #   (0, 14)       0.39654956741034536　・・・
 
test_pred = lg_model.predict_proba(X_test) #predict_proba: predict probability(予測確率)
# print(test_pred) ->
    # [[0.12204803 0.84510393 0.00923596 0.02361208]
    #  [0.16592815 0.73008149 0.02990813 0.07408223]
    #  ...
    #  [0.44186239 0.48329597 0.02926447 0.04557717]]
# print(lg_model.predict(X_test)) -> 
    #  ['e' 'e' 'b' ... 'e' 'b' 'e']




 #scikit-learnで混同行列を生成、適合率・再現率・F1値などを算出 https://note.nkmk.me/python-sklearn-confusion-matrix-score/
### 54 正解率(accuracy)の計測###
# 52で学習したロジスティック回帰モデルの正解率を，学習データおよび評価データ上で計測せよ．
from sklearn.metrics import accuracy_score # accuracy_score(y_true, y_pred)
accuracy_train = accuracy_score(df_train['CATEGORY'], lg_model.predict(X_train))
accuracy_test = accuracy_score(df_test['CATEGORY'], lg_model.predict(X_test))
print(f'正解率（学習データ）：{accuracy_train:.3f}')
print(f'正解率（評価データ）：{accuracy_test:.3f}')

### 55 混同行列の作成 ###
# 52で学習したロジスティック回帰モデルの混同行列（confusion matrix）を，学習データおよび評価データ上で作成せよ．
from sklearn.metrics import confusion_matrix # confusion_matrix(y_true, y_pred)
cm_train = confusion_matrix(df_train['CATEGORY'], lg_model.predict(X_train))
cm_test = confusion_matrix(df_test['CATEGORY'], lg_model.predict(X_test))
print(f"cm_test:\n {cm_test}")

#混同行列の可視化  1 import seaborn as sns / 2 sns.heatmap(confusion_matrix, annot=True, cmap = 'Blues)
import seaborn as sns
import matplotlib.pyplot as plt

#figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()

#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
sns.heatmap(cm_train, 
            annot=True, #セルに値出力
            cmap = 'Blues', #カラーマップの色
            ax = ax1
            )
sns.heatmap(cm_test, 
            annot=True, #セルに値出力
            cmap = 'Blues', #カラーマップの色
            ax = ax2
            )
plt.show()


### 56. 適合率，再現率，F1スコアの計測 ###
# 52で学習したロジスティック回帰モデルの適合率，再現率，F1スコアを，評価データ上で計測せよ．
# カテゴリごとに適合率，再現率，F1スコアを求めよ.
# カテゴリごとの性能をマイクロ平均（micro-average）とマクロ平均（macro-average）で統合せよ．

# ~ (1) 2クラス分類問題における4つの評価指標~ #　https://atmarkit.itmedia.co.jp/ait/articles/2210/24/news034.html
# 正解率（accuracy）: accuracy_score(y_true, y_pred)
# 適合率 (precision) : precision_score(y_true, y_pred)
# 再現率（recall）: recall_score(y_true, y_pred)
# F1値,F値（F1-measure）: f1_score(y_true, y_pred)

    #正解率 = (TP+TN) / (全体)
    #適合率 = TP / (TP+FP) =FP(偽陽性)を抑えたいときの指標
    #再現率 = TP / (TP+FN) =FN(偽陰性)を抑えたいときの指標
    #F1値 = 適合率と再現率の調和平均 = FN,FPをバランスよく抑えたい

# (2) 多クラス分類の評価指標# https://atmarkit.itmedia.co.jp/ait/articles/2212/19/news020.html
# マクロ平均
# マイクロ平均

# 二値分類の混同行列を作成するためのconfusion_matrixでもこの例のように４種に分類。（下の記事だと３種に分類してるのはどういうこと？）
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

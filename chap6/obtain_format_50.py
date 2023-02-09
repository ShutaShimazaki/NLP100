# URL: https://amaru-ai.com/entry/2022/10/12/202559
# 50 データの入手・整形
 #パス指定時に\ →\\にする。\だとエスケープ文字と認識されてしまうため
 #Pythonの外部ライブラリがVSCodeでcould not be resolvedとなる時の対処法→https://maasaablog.com/integrated-development-environment/visual-studio-code/4437/
 #pip install scikit-learn
 #組み込み関数、モジュール、パッケージ　https://www.wakuwakubank.com/posts/260-python-basic-function/
 #モジュール化⇒OOP　https://takun-physics.net/12376/



import sys,os
# sys.path.append('モジュールをインストールしたディレクトリ')
sys.path.append('C:\\Users\\user.DESKTOP-JAJ50S0\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\Scripts')

#https://pg-chain.com/python-zipfile-2
#https://tonari-it.com/windows-vscode-copy-relative-path/ (相対パスコピー時に、バックスラッシュをスラッシュに変換してくれる)
import zipfile
zip_f = zipfile.ZipFile("chap6\\NewsAggregatorDataset\\NewsAggregatorDataset.zip")
zip_f.extractall("chap6\\NewsAggregatorDataset")

import pandas as pd
from sklearn.model_selection import train_test_split

# データの読込
    #なにも引数を設定しないと、1行目がheaderとして認識され、列名columnsに割り当てられる。 / header=Noneとすると連番が列名columnsに割り当てられる。
    #names=('A', 'B', 'C', 'D')のように任意の値を列名として設定することもできる。リストやタプルで指定する。
    # https://note.nkmk.me/python-pandas-read-csv-tsv/
df = pd.read_csv('chap6\\NewsAggregatorDataset\\newsCorpora.csv', header=None, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
# print(len(df)) #422419 (本当は422937行のcsvファイルなのに)

#　データの抽出
df = df.query('PUBLISHER in ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]')
# stratifyで指定したものが均等になるようにdfを分割する
df_train, df_valid_test = train_test_split(df,test_size=0.2, shuffle=True, stratify=df["CATEGORY"]) 
df_valid, df_test = train_test_split(df_valid_test ,test_size=0.5, shuffle=True, stratify=df_valid_test["CATEGORY"]) 
    #df: (13340, 8)⇒df_train: (10672, 8) / df_valid: (1334, 8) / df_test: (1334, 8)

print(df_train['CATEGORY'].value_counts())
print(df_valid['CATEGORY'].value_counts())
print(df_test['CATEGORY'].value_counts())

# Dataframe→csv : df.to_csv
 # index(行名), header(列名)
df_train.to_csv('chap6\\csv\\df_train.txt', sep="\t", index = False)
df_valid.to_csv('chap6\\csv\\df_valid.txt', sep="\t", index = False)
df_test.to_csv('chap6\\csv\\df_test.txt', sep="\t", index = False)
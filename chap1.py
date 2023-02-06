#https://amaru-ai.com/entry/2022/10/05/090639#05-n-gram
#splitは分割後、リストとして取得

#00.文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．
#[開始位置:終了位置:移動幅]
#print('hello'[: : 2]) "hlo"
#print('hello'[: : 2]) "olh"
print("stressed"[::-1])

#01
string = "パタトクカシーー"
ans1 = string[::2]
print(ans1)

#02
string1 = "パトカー"
string2 = "タクシー"
#リスト生成：[式 for 任意の変数名 in イテラブルオブジェクト]
ans2 = "".join([i + j for i, j in zip(string1, string2)])
print(ans2)
#print([i + j for i, j in zip(string1, string2)]) ['パタ', 'トク', 'カシ', 'ーー']

#03
sentence3 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
#s.split(A) sをAで分割する。引数Aの省略時は、空白文字で区切る。
new_sentence3 = sentence3.replace(",","").replace(".","")
print(new_sentence3)
print([len(i) for i in new_sentence3.split()])

#05 n-gram :連続するn個の単語へ分割
# def n_gram(txt: str, n: int) -> list:
#     ans5_list = []
#     for i in range(len(txt)):
#         if i+n <= len(txt):
#             ans5_list.append(txt[i : i+n])
#     return ans5_list
# sentence5= 'I am an NLPer'
# print(n_gram(sentence5.split(), 2))
# print(n_gram(sentence5, 2))
# print(len(sentence5))

def ngram(n, lst):
  # ex.
  # [str[i:] for i in range(2)] -> ['I am an NLPer', ' am an NLPer']
  # zip(*[str[i:] for i in range(2)]) -> zip('I am an NLPer', ' am an NLPer')
  print(lst)
  print("1 ", *[lst[i:] for i in range(n)])
  print("1.5 ", [lst[i:] for i in range(n)])
  print("2 ", list(zip(*[lst[i:] for i in range(n)])))
  #zip関数＝２つのリスト、タプル、辞書をくっつける　https://camp.trainocate.co.jp/magazine/python-zip/
  return list(zip(*[lst[i:] for i in range(n)]))

str = 'I am an NLPer'

words_bi_gram = ngram(2, str.split())
# chars_bi_gram = ngram(2, str)

print('単語bi-gram:', words_bi_gram)
# print('文字bi-gram:', chars_bi_gram)

#06
def ngram_retry(txt, n):
  return list((zip(*[txt[i:] for i in range(n)])))
txt1 = 'paraparaparadise'
txt2 = 'paragraph'
bigram_txt1 = set(ngram_retry(txt1, 2))
bigram_txt2 = set(ngram_retry(txt2, 2))
sum_set = bigram_txt1 | bigram_txt2
difference_set = bigram_txt1 - bigram_txt2
product_set = bigram_txt1 & bigram_txt2
print("和集合 ", sum_set)
print("差集合 ", difference_set)
print("積集合 ", product_set)

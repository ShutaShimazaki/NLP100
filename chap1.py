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
print(sentence3.split())

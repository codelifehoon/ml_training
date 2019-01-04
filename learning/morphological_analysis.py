from konlpy.tag import  Okt,Kkma,Komoran



line  = "01.스트라이프배색 트임 면 긴팔티셔츠 / LMTS7A222W3_DARK BROWN:095 COLOR SIZE 색상사이즈"
# line  = "COLOR SIZE 색상사이즈"
twitter = Okt()
kkma = Kkma()
komoran = Komoran()


r = []

malist = twitter.pos(line,norm=True,stem=True)
for (word,pumsa) in malist:
    if not pumsa in ["Josa","Eomi","Punctuation"]:
        r.append(word)
        r.append(pumsa)

print(r)

print( twitter.nouns(line))

"""
import time
from konlpy.tag import Kkma, Okt, Komoran
pos_taggers = [('kkma', Kkma()), ('twitter', Okt()), ('Komoran', Komoran())]
results = []
for name, tagger in pos_taggers:
    tokens = []
    process_time = time.time()
    for text in texts:
        tokens.append(tagger.pos(text))
    process_time = time.time() - process_time
    print('tagger name = %10s, %.3f secs' % (name, process_time))
    results.append(tokens)
"""

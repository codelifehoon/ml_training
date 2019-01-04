import time
from konlpy.tag import Kkma, Okt, Komoran
pos_taggers = [('kkma', Kkma()), ('twitter', Okt()), ('Komoran', Komoran())]
results = []
for name, tagger in pos_taggers:
    tokens = []
    process_time = time.time()

    tokens.append(tagger.pos("COLORSIZE"))
    process_time = time.time() - process_time
    print('tagger name = %10s, %.3f secs' % (name, process_time))
    print(tokens)
    results.append(tokens)


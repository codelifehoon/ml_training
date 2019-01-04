import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
np.random.seed(0)

from konlpy.tag import Okt
twitter = Okt()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def tokenizer(raw, pos=["Noun","Alpha","Verb","Number"], stopword=[]):
    return [
        word for word, tag in twitter.pos(
            raw,
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
        )
        if len(word) > 1 and tag in pos and word not in stopword
    ]

# 테스트 문장
rawdata = [
    '남북 고위급회담 대표단 확정..남북 해빙모드 급물살',
    '[남북 고위급 회담]장차관만 6명..판 커지는 올림픽 회담',

    '문재인 대통령과 대통령의 영부인 김정숙여사 내외의 동반 1987 관람 후 인터뷰',
    '1987 본 문 대통령.."그런다고 바뀌나? 함께 하면 바뀐다"',

    '이명박 전 대통령과 전 대통령의 부인 김윤옥 여사, 그리고 전 대통령의 아들 이시형씨의 동반 검찰출석이 기대됨'
]


vectorize = CountVectorizer(
    tokenizer=tokenizer,
    min_df=2    # 예제로 보기 좋게 1번 정도만 노출되는 단어들은 무시하기로 했다
    # min_df = 0.01 : 문서의 1% 미만으로 나타나는 단어 무시
    # min_df = 10 : 문서에 10개 미만으로 나타나는 단어 무시
    # max_df = 0.80 : 문서의 80% 이상에 나타나는 단어 무시
    # max_df = 10 : 10개 이상의 문서에 나타나는 단어 무시
)

# 문장에서 노출되는 feature(특징이 될만한 단어) 수를 합한 Document Term Matrix(이하 DTM) 을 리턴한다
X = vectorize.fit_transform(rawdata)

print(
    'fit_transform, (sentence {}, feature {})'.format(X.shape[0], X.shape[1])
)
# fit_transform, (sentence 5, feature 7)

print(type(X))
# <class 'scipy.sparse.csr.csr_matrix'>

print(X.toarray())

# [[0, 1, 2, 0, 0, 0, 1],
# [0, 1, 1, 0, 0, 0, 2],
# [1, 0, 0, 2, 1, 1, 0],
# [1, 0, 0, 1, 0, 0, 0],
# [0, 0, 0, 3, 1, 1, 0]]

# 문장에서 뽑아낸 feature 들의 배열
features = vectorize.get_feature_names()
print(features)

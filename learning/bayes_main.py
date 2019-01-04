from konlpy.tag import Komoran
from bayesianFilter import BayesianFilter

komoran = Komoran()
bf = BayesianFilter()

# 텍스트 학습


bf.fit("size 사이즈 크기 90 95 100 105 110 115 095", "사이즈")
bf.fit("색상 색깔 color 빨강 노랑 파랑 빨간색 노란색 파란색 DARK BROWN","색상")


# 예측
value = "DARK"
pre, scorelist = bf.predict(value)
print("결과1 =", pre,scorelist)

value = "필수선택"
pre, scorelist = bf.predict(value)
print("결과2 =", pre,scorelist)

#
# print(komoran.morphs(u'우왕 코모란도 오픈소스가 되었어요'))
# print(komoran.nouns(u'오픈소스에 관심 많은 멋진 개발자님들!'))
# print(komoran.pos(u'한글형태소분석기 코모란 테스트 중 입니다.'))
# print(komoran.pos(u'사이즈/size/크기'))
# print(komoran.pos(u'사이즈size크기'))
# print(komoran.pos(u'사이즈,색상'))
# print(komoran.pos(u'사이즈색상color/size'))
# print(komoran.pos(u'A03_페도라 / 화이트 XL'))

"""
학습시작
자료 형태
    옵션명1,... 옵션명N ,
    stock_no,옵션값1,... 옵션값N
자료추출
    - 옵션명N classifier
        1. 옵션명N을 형태소 분석으로 분류
            ex : 색상  -> 하나의 명사로 구분
                 색상.사이즈 -> 두개의 색상,사이즈로 명사로 구분

        2. 분류된 옵션명으로  classifier 하여 옵션값이 어떠한 값인지 define
            ex:  색상,color,색깔 -> 색상
                 size,크기,사이즈 -> 사이즈

        2-1. 구분이 어려울 경우 옵션값N classifier를 통한 속성 구분
            ex : 250 -> 사이즈
                 빨강  -> 색상

    - 옵션값 classifier
        1. 옵션값N 대표값으로 classifier
            ex : 옐로, yellow,노랑 ->   노
                 250 -> 250



색상 , 사이즈 표준 테이블 준비
색상
C1        빨
C2        노
C3        파
사이즈
S1        240
S2        250
S3        260





옵션명,옵션값을 형태소 분류
사이즈 -> 사이즈
사이즈/색상 -> 사이즈 색상
Size/Color -> Size Color

분리된 문장을 vector로 변환
유사값을 text2vec 사용해도 될것 같은데..

vertor된 옵션값,옵션명을 classifier (사이즈,색상,ETC)

학습종료


학습 검증

"""
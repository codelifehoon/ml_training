import pandas as pd
import  numpy as np
import pprint as pp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups




pd.set_option('display.expand_frame_repr', False)

s = pd.Series([1, 3, 5, np.nan, 6, 8])
dates = pd.date_range('20130101', periods=6)


df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list(['AA','BB','CC','DD']))
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(3)), dtype='float32'),
                    'D': np.array([3]*3, dtype='int32'),
                    'E': pd.Categorical(['test', 'train', 'test']),
                    'F': 'foo'})



df_copy = df.copy()
df_minus_copy = df.copy()
df_copy['EE'] =  ['one', 'one','two','three','four','three']
df_minus_copy[df_minus_copy > 0] = -df_minus_copy


ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()



# df_grp = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
# df_grp = df_grp.cumsum()
# plt.figure(); df_grp.plot();

# plt.show()

# print(df)
# print(df.head(1))
# print(df.tail(1))
# print(df.T)
# print(df.sort_values(by='BB'))
# print(df.BB)
# print(df[0:2])
# print(df['2013-01-01':'2013-01-02'])
# print(df_minus_copy)



# print(df.loc[dates[0]])
# print(df.loc['2013-01-01'])
# print(df.loc[:,['AA','BB']])
# print(df.iloc[0])
# print(df.iloc[3:5,[0,2]])
# print(df.iloc[1:3,:])
# print(df_copy)
# print(df_copy['EE'].isin(['two', 'four']))


#
# print(df.index)
# print(df.columns)
# print(df.values)
#
#
#
#
# print(s)
# print(dates)
# print(df)
# print(df2)
# print(df.dtypes)
# print(df2.dtypes)


lists=[[3.64306109,7.95727485, 9.64993228, 2.98809026]
        ,[1.33306927, 4.50342279 , 1.48490343 , 1.44944056]
       ]

lists_df = pd.DataFrame(data=lists,columns=list(['A1','A2','A3','A4']))


print(lists)
print(lists_df)


news = fetch_20newsgroups(subset="test")
news_df = pd.DataFrame({'data':news.data,'target':news.target})
print(news_df)


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(news_df.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_)




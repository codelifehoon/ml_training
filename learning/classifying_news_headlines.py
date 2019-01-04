import pandas as pd
import pprint as pp
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.expand_frame_repr', False)
news = pd.read_csv('uci-news-aggregator.csv').sample(frac=0.1)


pp.pprint(news.head(3))




import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
import numpy as np
import  os
import pandas as pd
import pprint as pp
from konlpy.tag import  Okt
import pickle
import sys
import datetime


file_name = '/Users/codelife/Developer/11st_escrow2/74.pas-pay-kafkaproject/src/test/java/com/skp/payment/pas/tgcorp/repository/CardInfo.java'
with open(file_name,'r', encoding='utf-8',errors='ignore') as r:
    document = ''.join(r.readlines())
    print(document)



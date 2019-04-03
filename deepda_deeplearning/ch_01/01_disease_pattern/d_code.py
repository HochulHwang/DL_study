import codecs
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
import pandas as pd

#질병-가격 기록읽어오기 
dp_code = pd.read_csv('d_code.csv')
d_code = list(dp_code.d_code)
dp_code.head() 

#30대 질병데이터 읽어오기 
df = pd.read_table('yang_30.txt', sep='\n' , header = None)
df.columns = ['text']
df.head() 

#질병평균 가격표
dp_code_dic = dict(zip(dp_code.d_code , dp_code.price))

all_text = ''
for str in df.text:
        all_text = all_text + ' ' + str

text = all_text
print('훈련 데이터: ', len(text))

# 질병사전 구축하기 
text_split = text.split(" ")
chars = sorted(list(set(text_split)))
print('질병코드의 종류:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))  
indices_char = dict((i, c) for i, c in enumerate(chars))  

# 전체 훈련 데이터를 쪼개어서 문장구조로 만들기
maxlen = 5
step = 1
sentences = []
next_chars = []

for i in range(0, len(text_split) - maxlen, step):
    sentences.append(" ".join(text_split[i: i + maxlen]))
    next_chars.append(text_split[i + maxlen])
    
print('학습할 질병패턴의 수:', len(sentences))
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence.split(" ")):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 모델 구축하기(LSTM)
print('LSTM 모델을 만들어 본다....')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.015)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 후보추출
def sample(preds, temp=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
	
#질병코드 단어생성하기 
generated_all = []
for iteration in range(1, 8):
    print()
    print('반복번호 =', iteration)
    model.fit(X, y, batch_size=128, epochs=1)
    
    # 옵션을 다르게 하여 다양한 방법으로 생성하기
    for diversity in [1.4 , 1.6 , 1.8 , 2.0 , 2.2 , 2.4 , 2.6]:
        print()
        print('추출옵션 : ', diversity)
        generated = ''
        #sentence = text[start_index: start_index + maxlen]
        #sentence = "OL766 KG504 SF108"
        sentence = "S FG391"
        generated += sentence
        print('질병시작패턴 = "' + sentence + '"')
        sys.stdout.write(generated)
        
        # 질병코드 자동생성
        for i in range(35):      #문장의 평균 단어의 수(1년동안 질병의 코드의 수) 
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence.split(" ")):
                x[0, t, char_indices[char]] = 1.
                
            # 다음에 올 질병코드를 예측하기
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            
            # 30대 예상질병코드 출력하기
            generated += " " + next_char
            sentence = " ".join(sentence.split(" ")[1:]) + " " + next_char
            sys.stdout.write(" " + next_char)
            sys.stdout.flush()
            
            if next_char == "E":
                generated_all.append(generated)
                break
        print()

avg_price_1year = 0
sum_price = 0 
for i , sent in enumerate(generated_all):
    print(i,sent)
    for str in sent.split(" ")[1:-1]:
        sum_price += dp_code_dic[str]
        
avg_price_1year = round(sum_price / (i+1) , -2) 

print('-'*100)
print('이 분은 앞으로 1년간 총',avg_price_1year,'원의 병원비가 들 예정입니다.')
print('-'*100)
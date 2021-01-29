# 네트워크 분석 - Python

 텍스트 분석을 위한 여러 패키지들임. 아래 git에서 가져온 kospacing은 띄어쓰기를 해주는 패키지인데 안깔림. 왜인지는 잘 모르겠음
 Komoran은 한글 형태소 분석기인데 형태소 분석기마다 결과가 약간씩 다름. 이 분석에서는 설치 안해도 됨

```
 pip install wordcloud   
 pip install customized_konlpy   
 !pip install git+https://github.com/haven-jeon/PyKoSpacing.git   
 !pip install -U PyKomoran   
 !pip3 show nltk   
 !pip install apriori apyori   
 nltk.download('punkt')   
```
```
 import pandas as pd   
 import numpy as np   
```

파일 불러옴. 엑셀에서 csv를 저장하였을 경우 인코딩 문제로 불러지지 않는 경우가 있음. 이 경우 encoding='cp949' 를 넣어주면 잘 불러와짐

```
open1=pd.read_csv('C:/Users/CI/hyundai/youtube_raw.csv', sep=',', encoding='cp949')
```

댓글들을 모델별로 따로 분석해야 했기 때문에 미리 데이터를 모델별로 정의해 줌

```
deep_raw = open1['type'] == 1   
reply_raw = open1['type'] == 2   

reply_Tucson = open1['model'] ==1   
reply_Canival = open1['model'] ==2   
reply_Avante = open1['model'] ==3   
reply_G80 = open1['model'] ==4   
reply_Sorento = open1['model'] ==5   
reply_Sonata = open1['model'] ==6   
reply_Hyundai = open1['model'] ==7   
reply_Kia = open1['model'] ==8   
reply_Genesis = open1['model'] ==9   

subset_reply = open1[open1['type'] == 2]   

subset_Tucson = open1[reply_raw & reply_Tucson]   
subset_Canival = open1[reply_raw & reply_Canival]   
subset_Avante = open1[reply_raw & reply_Avante]   
subset_G80 = open1[reply_raw & reply_G80]   
subset_Sorento = open1[reply_raw & reply_Sorento]   
subset_Sonata = open1[reply_raw & reply_Sonata]   
subset_Hyundai = open1[reply_raw & reply_Hyundai]   
subset_Kia = open1[reply_raw & reply_Kia]   
subset_Genesis = open1[reply_raw & reply_Genesis]   


print(subset_Sonata)   
```

모델별 데이터에서 텍스트 자료가 있는 열만 뽑아옴

```
open2=subset_Sonata.iloc[:,4]   
```

데이터가 dataframe 형태일 경우 아래 apriori 패키지에서 에러가 발생하기 때문에 array 형태로 변경해 줌   
결측치 제거함   

```
lines=np.array(open2.dropna())   
print(lines)   
```

형태소 분석을 위한 패키지들을 불러옴   
Twitter를 쓴 이유는 사용자 사전을 추가하기가 편해서임   
(사실 다른 형태소 분석기들의 방법들도 찾긴 했지만 적용하는데 실패했음 ㅠㅠ)   

```
from ckonlpy.tag import Twitter   
from ckonlpy.utils import load_wordset   
from ckonlpy.utils import load_replace_wordpair   
from ckonlpy.utils import load_ngram   
from ckonlpy.tag import Postprocessor   

twitter=Twitter()   
```


사용자 사전을 파일 형태로 업데이트 하는 방법에 대해 더 알아봐야 함   
다양한 오타가 있을 수 있기 때문에 그랜져, 그랜저, 그렌저 등과 같이 동일 의미의 단어를 모두 잡아 주고 뒤의 replace 파일에서 한 단어로 모두 바꾸어줌   
모든 알파벳을 대문자 혹은 소문자로 변경하는 작업을 먼저 해주면 더 간단히 작업할 수 있음.    
소문자가 다른 의미를 가질 경우가 아니면 먼저 해주는 것이 좋음   

```
twitter.add_dictionary('승차감', 'Noun')   
twitter.add_dictionary('가성비', 'Noun')   
twitter.add_dictionary('g80', 'Noun')   
twitter.add_dictionary('G80', 'Noun')   
twitter.add_dictionary('g70', 'Noun')   
twitter.add_dictionary('G70', 'Noun')   
twitter.add_dictionary('GV80', 'Noun')    
#etc...  
```

stopwords: 빼야 할 단어   
replacewords: 바꿀 단어   
  바뀌기 전 단어 + tab + 바뀔 단어 순으로 한줄에 하나씩 적으면 됨   
ngrams: 함께 나왔을 때 합칠 단어   
  준+ spacebar + 중형+ tab +  준중형 -> 준 중형이 같이 나오면 '준 - 중형' 으로 표시됨. 그런데 안먹히는 경우 있음. 원인은 잘 모르겠음   

```
stopwords = load_wordset('C:/Users/CI/hyundai/stopwords.txt')   
replace = load_replace_wordpair('C:/Users/CI/hyundai/replacewords.txt')   
ngrams = load_ngram('C:/Users/CI/hyundai/ngrams.txt')   
```

Postprocessor 패키지를 사용하여 위에서 설정한 파일 적용   
password를 설정할 경우 여기에 속한 단어만 뽑혀나옴. 주의!!   
여기에서 정의한 단어 세트를 사용하여 기존 twitter 형태소 분석기를 업데이트 하는 것이라고 생각하면 됨   

```
pp1 = Postprocessor(twitter, stopwords=stopwords,
                    # passwords = passwords,
                    ngrams = ngrams,
                    replace = replace)
```

텍스트 자료를 위 형태소 분석기를 사용하여 토큰화함   
apriori 패키지를 사용할 수 있는 데이터 형태로 바꾸어야 함   
각 라인을 토큰화시켜 리스트로 만들고, 이 리스트들을 모아 또다른 리스트 안에 넣어줌   
[['a','b','c'],['d','c','e'],['a','f','g','d']] 이러한 형태가 되어야 함   
워드클라우드 생성을 위한 데이터를 생성할 때는 이러한 형태가 아니라 모든 데이터가 하나의 리스트 내에 들어가야 counter 패키지를 사용하여 빈도를 쉽게 뽑아낼 수 있음   
['a','b','c','d','c','e','a','f','g','d'] 이런 형태로 만들어 주어야 함   

for 문을 사용하여 각 라인의 댓글들을 토큰화시키고 이를 리스트로 만들어 또다른 리스트에 append 해줌   

```
word_tokens=[]

for i in range(len(lines)):
    word_tokens.append(pp1.pos(lines[i]))
```

twitter, Postprocessor 패키지를 사용하여 토큰화를 하면 [('자동차', 'Noun'), ('좋다','Verb')] 와 같이 품사가 같이 붙어서 데이터가 나옴   
이 중 필요한 명사, 동사, 형용사만 뽑아서 분석을 실시하였음.    
품사를 없애고 원 데이터만 남도록 데이터를 만듦   

```
word_pre = []

for item in word_tokens:
    temp = []
    for word, tag in item:
        if tag in ['Noun','Adjective', 'Verb'] and ("보다" not in word): # and ("쏘나타" not in word):
            temp.append(word)

    word_pre.append(temp)    

print(word_pre)
```

[['센슈어스', '옆 - 라인', '최고다'],   
 ['마력', '그랜저', '모델', '있었습니다'],   
 ['옆 - 라인', '만큼', '이쁘넹'],   
 ['냉각수', '붉', '은색', '녹나서', '붉어', '색']]   
 

이런 식으로 바뀜   
여기까지 하면 네트워크를 그리기 위한 준비가 끝난 것임   

--------------------------------------------

# 연관성 분석

노드의 크기 엣지의 굵기 및 길이를 설정하는 방법은 매우 많음   
아래 방법은 그 중 하나의 예에 불과함   
apriori 패키지를 사용하여 연관성 분석을 실시함   
연관성이 높은 단어 순으로 정렬해서 그 연관성을 edge의 굵기로 나타내줌   
단어들 간의 연관성이 너무 낮은 단어들은 잘라냄 (min_support=0.003)   


```
from apyori import apriori

results=list(apriori(word_pre,
                     min_support=0.003,
                     max_length=2))
```

연관성 분석 결과를 데이터프레임 형태로 만들어줌   
네트워크를 그릴 때 단어들이 너무 많으면 설정을 잘 하지 못하면 그림이 난잡해짐   
데이터를 살펴 보고 적정 단어 수를 결정함.   
support의 값을 변경하여 연관성이 낮은 단어부터 잘라냄   

```
df=pd.DataFrame(results)
df['length'] = df['items'].apply(lambda x:len(x))
df=df[(df['length']==2) & (df['support'] >=0.0052 ) ].sort_values(by='support', ascending=False)

df
```  

--------------------------------------------
  
# 네트워크 그리기

네트워크를 그리기 위한 패키지 불러옴   

```
import re
import networkx as nx
import matplotlib.pyplot as plt
```

위의 연관성 분석 후 남은 단어들을 각 노드로 정의해주고, networkx에서 사용할 수 있는 형태로 만들어 줌

```
G=nx.Graph()
ar=(df['items']); G.add_edges_from(ar)
```

각 단어들의 pagerank를 사용하여 각 노드의 크기를 결정해줌. 

```
pr=nx.pagerank(G)
nsize=np.array([v for v in pr.values()])
nsize=2000*(nsize-min(nsize))/(max(nsize)-min(nsize))
```

단어의 연관성(support)을 사용하여 edge의 굵기를 결정해줌. 0.001을 더해준 이유는 가장 얇은 선의 경우 보이지 않는 수준이라 전체적으로 굵게 만들어 준 것임

```
es = (df['support'])
width = 10*((es-min(es))+0.001)/(max(es)-min(es))
```

한글 폰트가 안 불러지는 경우가 많아서 아래와 같은 형태로 font_name이라는 변수에 원하는 폰트의 위치로 정의해줌

```
import matplotlib.font_manager as fm
from matplotlib import rc

font_name = fm.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
```


그림의 레이아웃을 설정해줌   
일반적으로 kamada_kawai가 가장 나은 모양을 보여줌.   
하지만 다른 단어들과 연관되지 않고 그들만의 연관성을 가진 단어조합이 있을 때 그림이 겹쳐보이면서 안보이는 경우가 있음   
이 경우 spring 레이아웃을 사용해 그 단어들을 따로 뽑아줌   
하지만 spring을 사용하면 노드들이 한군데로 많이 뭉쳐있어서 잘 안보이므로 그림 크기 자체를 크게 만들어서 그림 자체를 ppt내에서 편집해줌   
figsize=(40,30) 이 안의 숫자를 수정해줌 20, 15 정도면 충분하게 잘 보이지만 노드들이 너무 뭉쳐 있어서 그림을 크게 그리기 위해 40, 30으로 늘려준 것임   


```
#pos = nx.kamada_kawai_layout(G)
pos = nx.spring_layout(G)
#pos = nx.circular_layout(G)

plt.figure(figsize=(40,30)); plt.axis('off')
nx.draw_networkx(G,font_family=font_name, 
                 font_size=18,
                 pos=pos, 
                 width=width, 
                 node_color=list(pr.values()), 
                 node_size=nsize,
                 alpha=0.7, 
                 edge_color='.5', 
                 min_target_margin=20
                )
                
```


----------------------------------
# 워드클라우드 그리기

워드클라우드 그리기는 네트워크 분석에 비해 매우 간단함   
위에서 썼던 형태소 분석 결과를 그대로 가져와서 워드클라우드를 그림   
다만 데이터의 형태가 '리스트 내 리스트' 형태가 아니라 하나의 리스트 형태여야 함   
위에서의 설명 참조   

```
word_tokens2=[]

for i in range(len(lines)):
    word_tokens2.extend(pp1.pos(lines[i]))

```

아래 코드를 사용하여 사용할 품사들만 남기고 필요없는 단어들을 걸러줌
("보다" not in word) and ("아반떼" not in word) 이런 형태로 워드클라우드에서 보여지지 않을 단어 지정
다만 이때 '한' 이라는 단어를 걸러내기 위해서 ("한" not in word) 과 같은 형태를 쓴다면 '한'이 포함된 단어가 모두 사라지니 주의해야 함   
'한국', '강한', '한강' 등과 같은 단어가 모두 사라지게 됨    
이런 단어들은 토큰화 한 후에 걸러내는 것이 아니라 stopwords 에서 설정을 하여 형태소 분석 시 미리 걸러지도록 해야함   
   
이 분석의 경우 '아반떼','보다' 라는 단어가 네트워크 상에서 중요한 연결성을 가졌기 때문에 사용해야 했으나    
워드 클라우드에서는 '아반떼' 관련 댓글에 '아반떼'만 매우 큰 글자로 나오는 것이 너무 당연하므로 삭제를 한 것임   

```
cloud1 = []

for word, tag in word_tokens2:
    if tag in ['Noun', 'Adjective', 'Verb'] and ("보다" not in word) and ("아반떼" not in word):
        cloud1.append(word)
```

단어 토큰들의 빈도를 구해줌

```
from collections import Counter
count=Counter(cloud1)
words = dict(count.most_common())
```



워드클라우드 그려줌   
폰트, 그림 크기, 색 등을 설정할 수 있음   

```
from wordcloud import WordCloud 
import matplotlib 
import matplotlib.pyplot as plt 
from IPython.display import set_matplotlib_formats 

matplotlib.rc('font',family = 'Malgun Gothic') 
set_matplotlib_formats('retina') 
matplotlib.rc('axes',unicode_minus = False)

wordcloud = WordCloud(font_path = 'C:/Windows/Fonts/malgun.ttf', 
                      background_color='white',
                      colormap = "Accent_r", 
                      width=1500, height=1000).generate_from_frequencies(words) 
                      
plt.imshow(wordcloud) 
plt.axis('off') 
plt.show()

````




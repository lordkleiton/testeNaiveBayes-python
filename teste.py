import json
import nltk
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from pandas.io.json import json_normalize

##tweets
dataset = pd.read_csv('tweets_mg.csv')
dataset.count()

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

negativo = pd.read_json('negativo.json')

#for i in range(0, len(negativo)):
#    tweets.append(negativo["palavras"][i]["palavra"])

print(negativo["palavras"][1]["palavra"])
print(tweets[1])

a = negativo
a.append(tweets.tolist())

print(np.asarray(a))


vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

##textos recolhidos
textos = pd.read_json('textos.json')

##array a ser testado, se quiser colocar alguma frase pode colocar direto aí
testes = ['Esse governo está no início, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT'
        ]

##preenche o array que vai ser testado
for i in range(0, len(textos)):
    testes.append(textos["textos"][i]["texto"])

##tweets
freq_testes = vectorizer.transform(testes)
a = modelo.predict(freq_testes)

##print(list(a))
##print(a)

##cross validation
resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)

#print(list(resultados))
#print(resultados)
#print(len(resultados))

print('\nAcurácia: ', metrics.accuracy_score(classes,resultados))

##estatísticas gerais: precisão, revocação, pontuação f1
sentimento=['Positivo','Negativo','Neutro']
print('\nEstatísticas:\n', metrics.classification_report(classes,resultados,sentimento),'')

##matriz de confusão
print('\nMatriz de confusão:\n', pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')
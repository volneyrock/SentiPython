import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict


dataset = pd.read_csv('tweets.csv')

# Separa tweets das classificações
tweets = dataset['Text'].values
classi = dataset['Classificacao'].values


# Treina o modelo usando algorítimo Naive Bayes Multinomial #
# calcula a frequência de todas as palavras da lista de tweets
# vectorizer = CountVectorizer(analyzer='word')#Analiza palavra por palavra
vectorizer = CountVectorizer(ngram_range=(1,2))#Analiza pares de palavras
freq_tweets = vectorizer.fit_transform(tweets)
# Treina o modelo usando a frequencia de palavras e as classi de cada instância
modelo = MultinomialNB()
modelo.fit(freq_tweets, classi)


# Testes #
testes = ['Adorei o novo Hamburguer do Tucano',
         'A situação política do Brasil está uma droga, ninguém aguenta mais tamanha bandidagem',
         'O segundo filme do quarteto fantástico é uma porcaria.',
         'O estado de Santa Catarina decretou calamidade pública por conta das enchentes',
         ]

freq_testes = vectorizer.transform(testes)
# print(modelo.predict(freq_testes))
for i in testes:
    index = testes.index(i)
    print(i + ': >>>' + modelo.predict(freq_testes)[index])

# Validação do modelo
resultados = cross_val_predict(modelo, freq_tweets, classi, cv=10)
print("Acurácia: {0}%".format(str(metrics.accuracy_score(classi, resultados))[2:4]))
# Matriz de confusão
print('Matriz de confusão:')
print (pd.crosstab(classi, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')

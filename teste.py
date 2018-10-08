import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

dataset = pd.read_csv('tweets_mg.csv')
dataset.count()

tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

vectorizer = CountVectorizer(ngram_range=(1,2))
freq_tweets = vectorizer.fit_transform(tweets)
modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

testes = ['Esse governo está no início, vamos ver o que vai dar',
         'Estou muito feliz com o governo de Minas esse ano',
         'O estado de Minas Gerais decretou calamidade financeira!!!',
         'A segurança desse país está deixando a desejar',
         'O governador de Minas é do PT',
         'Boa noite a tod@s!!Estamos iniciando mais um fórum sobre  Papel e identidade de Gênero. Nossa reflexão principal é pensar como o conceito de gênero trouxe contribuições importantes para repensar o lugar das mulheres na família, no mercado de trabalho e na politica. É importante considerar que nas ultimas décadas  discussão de gênero tem evoluído quando reconhece que não se trata apenas de questões relacionadas as mulheres, mas dos múltiplos gêneros.  Essas mudanças , que fazem parte de uma determinada sociedade e cultura, exigem tempo e luta. As discussões que serão levantadas são de extrema  importância para a reflexão sobre questões complexas como o preconceito, o machismo, a homofobia, a violência conta as mulheres, que estão no nosso cotidiano, nos diversos espaços sociais que frequentamos. É importante enfatizar que o lugar social que historicamente foi destinado às mulheres é uma construção social. Essa construção social, que coloca a mulher como um ser inferior, é ainda hoje uma questão bastante problemática, levando muitas mulheres a viverem sobre opressão e sofrerem diversos tipos de violência. Não podemos esquecer que todos/as nós estamos inseridos/as nessa sociedade machista e, portanto, estamos todos/as sujeitos a essa lógica. Tivemos muitos avanços e muitas conquistas no que se refere à conquista por direitos iguais, mas ainda temos muito o que avançar, discutir, construir. Os movimentos sociais desempenham um papel fundamental na desconstrução dessa sociedade opressora e na construção de uma sociedade mais plural.'
         ,'todo político é ladrão',
         'Bom dia a tod@s!!!!         Ninguém nasce mulher: torna-se mulher. Nenhum destino biológico, psíquico, econômico define a forma que a fêmea humana assume no seio da sociedade; é o conjunto da civilização que elabora esse produto intermediário entre o macho e o castrado que qualificam de feminino. Somente a mediação de outrem pode constituir um indivíduo como um outro. (Simone de Beauvoir)          No cotidiano, vivenciamos inúmeros debates/opiniões a respeito de gênero. É importante salientar que  que muitas veze, revestido de preconceitos . De uma maneira errônea , muita gente acredita que o tema gênero diz respeito somente a” mulher”. É importante porém  ressaltar  que o tema gênero, não diz respeito somente as mulheres, mas também aos homens, uma vez que no remete a discussão sobre direitos humanos, diversidade e respeito ao outro/a.        A pesquisadora   Júlia de Arruda Rodrigue corrobora com nossa  leituras e aponta que “ podemos extrair que o gênero não é fruto da natureza, e sim uma construção social e histórica que atribui papeis a homens e mulheres com base nas diferenças entre os sexos biológicos, redundando na naturalização das discriminações contra o gênero feminino, em virtude de serem tomadas como decorrência inevitável das diferenças entre os sexos. Em outras palavras, existem relações de poder entre homens e mulheres, nas quais estas são tidas como inferiores, vez que seu gênero é construído a partir daquele que é dominante: o masculino”'
         , 'BOA VIAGEM/CE CIROU, amanhã é 1⃣2⃣!'
         ]

freq_testes = vectorizer.transform(testes)
print(modelo.predict(freq_testes))

resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)

print (resultados)

print(metrics.accuracy_score(classes,resultados))

sentimento=['Positivo','Negativo','Neutro']
print (metrics.classification_report(classes,resultados,sentimento),'')

print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True), '')


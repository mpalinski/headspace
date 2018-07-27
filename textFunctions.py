from unidecode import unidecode
import re
import string
import nltk.data
from langdetect import detect

## Przygotowanie tekstu
def langDetect(textArg):
    '''Zwraca kod języka. Działa w tempie ok. 6000 na minutę (testowane na opisach sylabusów).
    Argument: string

    Dla 'nan' zwracałoby tl, dla '' LangDetectException.
    '''

    text = str(textArg)
    if text == 'nan' or text == '':
        return('')
    else:
        return(detect(text))

def tokenize(textArg, bigrams=False, words=False, lang='polish'):
    '''Zwraca stokenizowany tekst. Należy zrobić przed dalszymi krokami, ponieważ niektóre z dalszych funkcji wykonują się rekurencyjnie i mogą działać źle/nieprzewidywalnie dla stringów.

    Argumenty:
    - textArg: string
    - words: True jeśli chcemy listy słów oddzielonych spacjami w tekście, False jeśli chcemy listy list słów oddzielonych spacjami
    - lang: język. W katalogu nltk_data: tokenizers/punkt lista dostępnych

    Wymaga punkt z nltk.data. Jeśli nie ma, pyta, czy chce pobrać.
    '''

    if words:
        tokens = textArg.strip().split(' ')
    else:
        try:
            tokenizer = nltk.data.load('tokenizers/punkt/' + lang.lower() + '.pickle') # tokenizer dla języka polskiego
        except LookupError: # brak tokenizera
            if str(input('Naciśnij Y, aby pobrać potrzebny pakiet: punkt (nltk)')) == 'Y':
                nltk.download('punkt') # pobieranie
                tokenizer = nltk.data.load('tokenizers/punkt/' + lang.lower() + '.pickle')
        sentences = tokenizer.tokenize(textArg.strip()) # sentences to lista zdań, w których słowa nie są oddzielone
        tokens = []
        for i in sentences:
            tokens.append(i.strip().split(' '))
            # tokens.append(tokenizer.tokenize(i.strip()))

    if bigrams:
        tokens_bigrams = []
        for i in range(0, len(tokens)-1):
            tokens_bigrams.append(tokens[i] + ' ' + tokens[i+1])
        return tokens_bigrams
    else:
        return tokens

def cleaning(tokensArg, 
        removePunctuation=False, 
        removeNumbers=False,
        toLower=False, 
        toLowerFirst=False,
        removeSpaces=False, 
        stopwords=False, 
        stopwordsUser=[],
        decodeDiacritics=False,
        shortwords=False):
    
    '''Zwraca wyczyszczone tokeny. Jest wiele argumentów, domyślna kolejność wydaje się logiczna dla większości zastosowań (aby można było wykonać cleaning jeden raz).

    Argumenty:
    - tokensArg: lista tokenów (użyj funkcji tokenize)
    - removePunctuation: usunięcie !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»…—
    - removeNumbers: usunięcie cyfr. Dla przykładu, 'a42b' zmienia na 'ab'
    - toLower: zamiana słów na małe litery
    - toLowerFirst: zamiana słów na małe litery, gdy pierwsze na liście stringów. Nie używać na jednopoziomowych listach. Może być przydatne do prostego znalezienia nazw własnych
    - removeSpaces: usunięcie nadmiarowych spacji
    - stopwords: usunięcie stopwords z listy zdefiniowanej w stopwords.txt: https://github.com/bieli/stopwords/blob/master/polish.stopwords.txt
    - stopwordsUser: lista stopwords zdefiniowanych przez użytkownika
    - decodeDiacritics: zamiana polskich znaków na litery występujące w alfabecie angielskim. stopwords może nie zadziałać poprawnie bez dodatkowo zdefiniowanej listy, jeśli wykonane zostanie cleaning z decodeDiacritics przed usunięciem stopwords
    - shortwords: usunięcie słów krótszych niż 3 litery (czyli 2 albo 1 litera). Wykonać po decodeDiacritics: regex nie uwzględnia polskich znaków, czyli straciłoby słowa "sąd", "koń", 'Łęcką', 'łóżko' itd.

    Usuwanie stopwordsUser wymaga stopwords=True.
    '''

    if stopwords: # wczytanie pliku ze stopwords i dodanie zdefiniowanych przez użytkownika
        with open('stopwords.txt') as file:
            stopwordsList = file.read().splitlines()
        stopwordsList.extend(stopwordsUser) 

    for i in range(0, len(tokensArg)):
        if type(tokensArg[i]) == list:
            tokensArg[i] = cleaning(tokensArg[i], 
                                removePunctuation=removePunctuation, 
                                removeNumbers=removeNumbers,
                                toLower=toLower, 
                                toLowerFirst=toLowerFirst,
                                removeSpaces=removeSpaces, 
                                stopwords=stopwords, 
                                stopwordsUser=stopwordsUser,
                                decodeDiacritics=decodeDiacritics,
                                shortwords=shortwords)

        elif type(tokensArg[i]) == str:
            if decodeDiacritics:
                tokensArg[i] = unidecode(tokensArg[i])
            if removePunctuation:
                punct = string.punctuation + '«»…—’‘“”' # kilka dodatkowych znaków
                tokensArg[i] = tokensArg[i].translate(str.maketrans('','',punct)) # usunięcie znaków ze stringa
            if removeNumbers:
                tokensArg[i] = re.sub('\d+','',tokensArg[i])
            if toLower:
                tokensArg[i] = tokensArg[i].lower()
            if (i == 0) and toLowerFirst:
                tokensArg[0] = tokensArg[0].lower()
            if removeSpaces:
                tokensArg[i] = re.sub(' +',' ',tokensArg[i])
            if stopwords:
                if tokensArg[i].lower() in stopwordsList: # działa również wtedy, gdy nie zmienimy na lowercase
                    tokensArg[i] = ''
            if shortwords:
                if len(re.findall('[a-zA-Z]', tokensArg[i])) < 3:
                    tokensArg[i] = ''

    tokensArg = [word for word in tokensArg if word != ''] # usuwamy stopwords i shortwords, które zamieniliśmy wcześniej na ''
    return tokensArg


## stemming
''' 
Podstawa: https://github.com/morfologik/polimorfologik

Na razie należy uważać ze stemmingiem. Zdecydowanie nie działa on dobrze i na oko wyniki, które są otrzymywane bez stemmingu są bardziej zgodne z rzeczywistością.

Słowa rozpoczynające się wielką literą są rozpoznawane jako nazwy własne. Dodatkowo stemming nie potrafi znaleźć tematu wyrazu, gdy jest on nazwą własną zapisaną małą literą (np. litwy).
'''
import os

def stems(tokensArg, directory='.', output='stems.tmp'):
    '''Zapisuje do tymczasowego pliku output javy morfologika.

    Argumenty:
    - tokensArg: lista tokenów (użyj funkcji tokenize)
    - directory: katalog z morfologikiem
    - output: plik, do którego zapisać
    '''
    if type(tokensArg[0]) == list:
        oneLvlTokens = [item for sublist in tokensArg for item in sublist] # jeśli lista ma 2 poziomy, to spłaszczamy
    else:
        oneLvlTokens = tokensArg
    wordsNewline = ''
    for i in set(oneLvlTokens): # zbiór, aby nie powtarzać słów
        wordsNewline += i.lower() + '\n'
        wordsNewline += i + '\n'
    with open('words.tmp', 'w') as file:
        file.write(wordsNewline) # morfologik potrzebuje pliku ze słowami oddzielonymi nową linią
    command = ['java', '-jar', directory + '/lib/morfologik-tools-2.1.0.jar', 'dict_apply', '-d', directory + '/polish.dict', '-i', 'words.tmp', '--input-charset', 'utf-8', '> ', output]
    os.system(' '.join(command))

def dictStems(stemFile='stems.tmp', stemsAsList=False):
    '''Zwraca słownik stemów.
    
    Argumenty:
    - stemFile: plik, z którego czyta output morfologika
    - stemsAsList: jeśli chcemy zachować wszystkie możliwe tematy (przejść -> [przejście, przejść], bo dopełniacz liczby mnogiej/bezokolicznik), można je zapisać jako listę. Jeśli False, bierze pierwszy możliwy 
    '''
    dictOfStems = {}
    with open(stemFile, 'r') as file:
        for i in file.readlines():
            # format to: przejść => przejść verb:inf:perf:nonrefl+verb:inf:perf:refl. Być może w przyszłości będziemy chcieli zachować informacje o słowie, ale na razie nie zapisujemy ich.
            orig = re.findall('(\w+) =>', i)
            stem = re.findall('=> (\w+) ', i)
            info = i.split(' ')[-1]
            if len(orig) > 0:
                if stemsAsList == True:
                    if orig[0] in dictOfStems.keys():
                        dictOfStems[orig[0]].extend(stem)
                    else:
                        dictOfStems[orig[0]] = stem
                else:
                    if len(stem) > 0:
                        dictOfStems[orig[0]] = stem[0]
                        dictOfStems[orig[0].capitalize()] = stem[0]
                    else:
                        if orig[0] not in dictOfStems.keys():
                            dictOfStems[orig[0]] = orig[0]
    return dictOfStems

def tokenStemming(tokensArg, dictOfStems):
    '''Zwraca stemy tokenów.

    Argumenty:
    - tokensArg: lista tokenów (użyj funkcji tokenize)
    - dictOfStems: słownik stemów w formacie token: stem (użyj funkcji dictStems)
    '''
    for i in range(0, len(tokensArg)):
        if type(tokensArg[i]) == list:
            tokensArg[i] = tokenStemming(tokensArg[i], dictOfStems)
        elif type(tokensArg[i]) == str:
            try:
                dictOfStems[tokensArg[i]] # jeśli ta linijka daje KeyError, nie wykonuje się następna i nie sprawia problemów z przypisaniem
                tokensArg[i] = dictOfStems[tokensArg[i]]
            except KeyError:
                pass
    return tokensArg

## Analiza tekstu
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

def vectorize(scikitVecArg='CountVectorizer', maxFeatures = 5000):
    '''Zwraca vectorizer.

    Argumenty:
    - scikitVec: string vectorizera do użycia. Dostępne: CountVectorizer (domyślne), TfidfVectorizer
    - maxFeatures: maksymalna liczba cech. Domyślnie 5000
    '''
    if scikitVecArg == 'TfidfVectorizer':
        scikitVec = TfidfVectorizer
    else:
        scikitVec = CountVectorizer
    vectorizer = scikitVec(analyzer = "word",
                        tokenizer = lambda doc: doc, # funkcje wyżej
                        lowercase = False, # funkcje wyżej
                        # Potrzebne tokenizer i lowercase, ponieważ możemy używać listy list
                        max_features = maxFeatures) 
    return vectorizer

def tokensToArray(vectorizer, tokensArg):
    '''Zwraca tablicę numpy z wektorami.

    Argumenty:
    - vectorizer: odpowiedni vectorizer (użyj funkcji vectorize)
    - tokensArg: lista tokenów (użyj funkcji tokenize)
    '''
    data_features = vectorizer.fit_transform(tokensArg)
    return data_features.toarray()

def mostUsedWords(vectorizer, tokensArg, number=20):
    '''Zwraca listę tupli najczęściej używanych słów. Tuple zawierają słowo i wartość z vectorizera.
    
    Argumenty:
    - vectorizer: odpowiedni vectorizer (użyj funkcji vectorize). Należy użyć CountVectorizer, ponieważ dostajemy liczbę wystąpień, a nie sumę normalizowanych wartości bez sensownej intepretacji (inne wyniki, choć podobne)
    - tokensArg: lista tokenów (użyj funkcji tokenize)
    - number: liczba słów, domyślnie 20
    '''
    data_features = vectorizer.fit_transform(tokensArg)
    data_features = data_features.toarray()
    dist = np.sum(data_features, axis=0) # suma słów
    vocab = vectorizer.get_feature_names()
    dict1 = {}
    for tag, count in zip(vocab, dist):
        dict1[tag] = count
    sortedWords = []
    # sortujemy
    for w in sorted(dict1, key=dict1.get, reverse=True):
        sortedWords.append((w, dict1[w]))
    return sortedWords[0:number]

## word2vec

from gensim.models import word2vec

def modelVec(tokensArg,
          num_features = 300,
          min_count = 5,
          num_workers = 4,
          context = 10,
          downsampling = 1e-3):
    '''Zwraca model word2vec.

    Argumenty:
    num_features: wymiar wektora słów
    min_word_count: ignoruj słowa występujące rzadziej
    num_workers: liczba threadów
    context: rozmiar okienka kontekstu
    downsampling: downsampling dla często występujących słów. Rekomendowane między 1e-5 a 1e-3.
    '''
    model = word2vec.Word2Vec(tokensArg, workers=num_workers,
                size=num_features, min_count = min_count,
                window = context, sample = downsampling)

    model.init_sims(replace=True)
    return model
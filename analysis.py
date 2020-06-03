from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import print_function

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import gc
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
import re
import sys
print(sys.path)
sys.path = set(sys.path)
#sys.path.append(r'C:\dev\ds-recsys')

import os
# os.chdir(r'C:\dev\ds-recsys')

from nltk.tokenize import sent_tokenize
import string
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
#from operator import itemgetter
import spacy
#nlp = spacy.load('en') # not working
import en_core_web_sm
nlp = en_core_web_sm.load() # works
from spacy.lang.en import English as nlp
from spacy.lang.en import STOP_WORDS as STOPWORDS
stoplist = set(stopwords.words('english'))
print(len(STOPWORDS)) #305

import nltk
nltk.download('stopwords')
stoplist = set(stopwords.words('english'))
print(len(stoplist)) # 179

# sentence tokenization
# pos
# chunking
# lemmatization
# stop words removal
# other filters - remove 1 charcater etc..



def normalize_text(text):
    sentence = text.lower()
    pt = tokenize_sentence_with_pos(sentence)
   
    word_pos = []
    for n in pt:
        if n[1] in tags_to_keep:   
            if re.match('^[0-9.,/]+', n[0].lower_) is not None:
                continue
            if n[0].lemma_ != "-PRON-":
                word_pos.append( (  n[0].lemma_.strip(), n[1])  )
            else:
                word_pos.append( (  n[0].lower_, n[1])  )
           
    words = [ w for w, tag in word_pos if w not in stoplist]
   
    words = [w.translate(table) for w in words]
    # add n-grams
    words.extend(get_ngrams(sentence))
   
    return [w for w in words if len(w) > 1]



def tokenize_document(doc):
    result = []
   
    for sentence in sent_tokenize(doc):
       
        result += normalize_text(sentence.strip())

    return result



table = str.maketrans('', '', '\'`')

punctuations = string.punctuation
# https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
tags_to_keep = ['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def get_spacy_token_text(token):
    text = ''
    if token.lemma_ != '-PRON-':
        text = token.lemma_.strip()
    else:
        text = token.lower_
    return text   

def get_ngrams(sentence):
    # spacy entities relies on upper case sentence so can't use it here
    # since input is already lowercase
    ngrams = []
    spacy_doc = nlp(sentence)

    # get all coniguous-NOUN
    # get all ADJ contigous-NOUN
    prev_token = ''
    noun_list = []
    for token in spacy_doc:
        # print(token, isinstance(token,spacy.tokens.token.Token ))
        # if isinstance(prev_token, spacy.tokens.token.Token):
        #     print(token, token.pos_, prev_token, prev_token.pos_)
        if token.pos_ == 'NOUN':
            noun_list.append(get_spacy_token_text(token))
        else:   
            if len(noun_list) > 0:
                contigous_noun = ' '.join(noun_list)
            ngram = ''   
            if isinstance(prev_token, spacy.tokens.token.Token) and prev_token.pos_ == 'ADJ':
                ngram = get_spacy_token_text(prev_token) + ' '
                #print(ngram)
            if len(noun_list) > 1 or (len(ngram) > 0 and len(noun_list) == 1):
                ngram +=  contigous_noun   
                ngrams.append(ngram) 
            noun_list=[]
            prev_token = ''
        if token.pos_ == 'ADJ' and token.lemma_ != '-PRON-':
            prev_token = token
       
    if len(noun_list) > 0:
        contigous_noun = ' '.join(noun_list)
        ngram = ''   
        if isinstance(prev_token, spacy.tokens.token.Token) and prev_token.pos_ == 'ADJ':
            ngram = get_spacy_token_text(prev_token) + ' '
        if len(noun_list) > 1 or (len(ngram) > 0 and len(noun_list) == 1):
            ngram +=  contigous_noun   
            ngrams.append(ngram)   
    return ngrams       
       
# get_ngrams(doc)   
   
def print_pos(sentence):
    sentence = sentence.lower()
    doc = nlp(sentence)
    for s in doc.sents:
        for token in s:
            print(token, token.pos_, token.tag_, token.ent_type_, token.lemma_)
   
   
def tokenize_sentence_with_pos(sentence):
    doc = nlp(sentence) 
    tokens = []   
    pt = [ (word, word.tag_) for word in list(doc.sents)[0] ]
    pt = [ (word, word.tag_) for sent in list(doc.sents) for word in sent ] 
    return pt




terminators = ['.', '!', '?']

def endswith(s, terminators):

    for t in terminators:
        if s.strip().endswith(t):
            return True
    return False

# return list of tuples
def create_documents(articles_list):
    documents = []
    for a in articles_list:
        document = ''
        index = -1
        link_id = 0
        for p in a:
            index += 1
            if index == 0:
                link_id = p
                continue
            terminator = '.'
            if len(p) > 0:
                if endswith(p, terminators):
                    terminator = ''
                document += (p + terminator + ' ')
        documents.append( (link_id, document) )       
    return documents       

import nltk               
def freqdist():
    # some stats
    fdist = nltk.FreqDist([ a[1] for a in articles_list])   
    print('max freq:', fdist.max(), '\n')
   
    print(fdist.most_common(30))

# based on freqdist, remove these domain specific stopwords
# ignore stopwords list:
domain_stop_words = [
#        ''
]

domain_stop_words_lower = [w.lower() for w in domain_stop_words ]
   
def has_domain_stop_words(document):
    # assume document has no trailing spaces
    if document.strip().lower() in domain_stop_words_lower:
        return True
    return False

# has_domain_stop_words('cookie consent msg etc.?')

def filter_domain_stopwords(documents):
    documents_no_stop = []
    for document in documents:
        doc = document[1].strip()
        tokens = nltk.word_
       
        (doc)
       
        if len(tokens) < 4: # period is included
            continue
        if has_domain_stop_words(doc):
            continue
        documents_no_stop.append( ( document[0], doc) )
    return documents_no_stop
   


def get_unique_documents(documents):
    #r eturn list(nltk.FreqDist(documents).keys())
   
    # return unique docments by text.  pick first link_id
    text_set = set()
    unique_documents = []
   
    for doc in documents:
        if not doc[1] in text_set:
            unique_documents.append(doc)
            text_set.add(doc[1])
           
    return unique_documents



def export_corpus_tfidf(dictionary, corpus_tfidf):
    n_items = len(dictionary)
    ds = []
    for doc in corpus_tfidf:
        d = [0] * n_items
    for index, value in doc :
        d[index]  = value
    ds.append(d)
    return ds   



def export_corpus_lsi(dictionary, corpus_lsi):
    n_items = lsi.num_topics
    ds = []
    for doc in corpus_lsi:
        d = [0] * n_items
        for index, value in doc :
            d[index]  = value
        ds.append(d)
    return ds   



def export_corpus_lda(dictionary, corpus_lda):
    n_items = lda.num_topics
    ds = []
    for doc in corpus_lda:
        d = [0] * n_items
        for index, value in doc :
            d[index]  = value
        ds.append(d)
    return ds 





import pandas as pd
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_colwidth', -1)


path = "/Users/lucui/NLP_projects/20200602_web_doc/data/url_text_w_id.csv"

brand='processing'

article_df = pd.read_csv(path, keep_default_na=False, sep='|', encoding="ISO-8859-1")


# label = [0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,2,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# df_label = pd.DataFrame(label, columns=['label'])
# concatenated = pd.concat([df_label, article_df], axis=1)
# print(concatenated[['label','text']])


article_df_copy = article_df.copy()

articles_list = article_df[['text']].values.tolist()
documents = create_documents(articles_list) #article_df.id.value_counts()


import nltk
nltk.download('punkt')
#domain_stop_words_lower = []
#documents_no_stopwords = filter_domain_stopwords(documents)
documents_no_stopwords = documents
#%time documents_unique = list(nltk.FreqDist(documents_no_stopwords).keys())
#documents_unique = get_unique_documents(documents_no_stopwords)
documents_unique = documents_no_stopwords
#vehicle_research_articles = [ d for d in documents_unique if 'some keyword' in d[0]]
#search_articles = [ d for d in vehicle_research_articles if 'some text' in d[1]]



def filter_pos_tags(pt):
    tokens = []
    for n in pt:
        #if not n[1] in ['CD', 'MD', 'IN', ':', ',' , 'HYPH', 'POS', '-LRB-', '-RRB-', '.', 'NFP']:
        #if n[1] in tags_to_keep:   
        if not n[1] in ['CD' , 'NFP', ':', ',']:
            if re.match('^[0-9.,/]+', n[0].lower_) is not None:
                continue
           
            if n[0].lemma_ != "-PRON-":
                tok = n[0].lemma_.strip().rstrip('-')
            else:
                tok = n[0].lower_.rstrip('-')
            tokens.append(tok)   
    return tokens







len(documents_unique) # 59

del documents
del documents_no_stopwords
del articles_list

import gc
gc.collect()


def tokenize_document(doc):
    result = []
    for sentence in sent_tokenize(doc):
        result += normalize_text(sentence.strip())
    return result




import time
start_time = time.clock()
#documents_unique = documents_unique[0:100]
#texts = [ tokenize_document(document) for link_id, document in documents_unique ]
# texts = [ tokenize_document(document) for document in documents_unique ]
texts = [ tokenize_document(document) for document, sep in documents_unique ]

print(time.clock() - start_time, "seconds")

import pickle
with open('/Users/lucui/NLP_projects/20200602_web_doc/data/file.pkl', 'wb') as fp:
    pickle.dump(texts, fp)
with open('/Users/lucui/NLP_projects/20200602_web_doc/data/file.pkl', 'rb') as fp:
    texts= pickle.load(fp)


dictionary = corpora.Dictionary(texts)
print(dictionary) #Dictionary(458 unique tokens

dictionary.save('/Users/lucui/NLP_projects/20200602_web_doc/' + brand + '-taxonomy.dict')  # store the dictionary, for future reference

corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus, normalize=True) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]

# get similar document for record [0]
index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=53413)
#and to query the similarity of our query vector vec against every document in the corpus:
# =============================================================================
# vec = df.iloc[:,0]
# vec = corpus[0]
# vec
# sims = index[tfidf[vec]]
# print(list(enumerate(sims)))
#
# =============================================================================

vocabulary =  [ dictionary.get(i) for i in range(len(dictionary.values()))]
len(vocabulary) # 458
ds = export_corpus_tfidf(dictionary, corpus_tfidf)

df = pd.DataFrame.from_records(ds)
df.columns = vocabulary
df.head()
lid = [typeofdoc for typeofdoc, text in documents_unique]
#article_df_p[['id','text']]
df['taxonomy'] = lid

print(df.shape)
df.head()
#df.to_csv(brand + '-web-articles-tfidf.csv', sep='\t', index=False)
df.to_csv(brand + '-tfidf.csv', sep='\t', index=False)

df = df.drop(corpus_tfidf['taxonomy'])


np.random.seed(1) # setting random seed to get the same results each time.


from gensim.models import lsimodel

for n in [5, 10]:
#below 2 are the same
# lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=500)
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n)
    #https://radimrehurek.com/gensim/models/ldamodel.html
    #lsi = lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
    # using corpus or corpus_tfidf does not seem to change topics
    corpus_lsi = lsi[corpus_tfidf]
    #corpus_lsi = lsi[corpus] 
    vocabulary =  [ dictionary.get(i) for i in range(len(dictionary.values()))]
    lsi_ds = export_corpus_lsi(dictionary, corpus_lsi)
    lsi_df = pd.DataFrame.from_records(lsi_ds)
    #lsi_df.columns = vocabulary
    lid = [typeofdoc for typeofdoc, text in documents_unique]
#    lsi_df['taxonomy'] = lid
    print(lsi_df.shape)
    #lsi_df.to_csv(brand + '-lsi.csv', sep='\t', index=False)
    lsi_df.to_csv(brand + '-lsi-' + str(n) + '.csv', sep='\t', index=False)

"""
2020-06-03 03:51:03,026 : INFO : using serial LSI version on this node
2020-06-03 03:51:03,027 : INFO : updating model with new documents
2020-06-03 03:51:03,037 : INFO : preparing a new chunk of documents
2020-06-03 03:51:03,038 : INFO : using 100 extra samples and 2 power iterations
2020-06-03 03:51:03,038 : INFO : 1st phase: constructing (458, 105) action matrix
2020-06-03 03:51:03,039 : INFO : orthonormalizing (458, 105) action matrix
2020-06-03 03:51:03,047 : INFO : 2nd phase: running dense svd on (105, 59) matrix
2020-06-03 03:51:03,050 : INFO : computing the final decomposition
2020-06-03 03:51:03,051 : INFO : keeping 5 factors (discarding 68.538% of energy spectrum)
2020-06-03 03:51:03,051 : INFO : processed documents up to #59
2020-06-03 03:51:03,052 : INFO : topic #0(2.120): -0.325*"city" + -0.172*"floyd" + -0.172*"george" + -0.164*"refuse" + -0.164*"thousand" + -0.164*"city official" + -0.164*"afternoon" + -0.164*"distinguish" + -0.164*"honor" + -0.164*"arrest"
2020-06-03 03:51:03,052 : INFO : topic #1(1.977): -0.212*"city" + -0.204*"widespread destruction" + -0.204*"clash" + -0.204*"destruction" + -0.204*"instituting" + -0.204*"instituting curfew" + -0.204*"looter" + -0.204*"weekend" + -0.204*"widespread" + -0.204*"cause"
2020-06-03 03:51:03,052 : INFO : topic #2(1.910): -0.157*"silicon" + -0.157*"valley" + -0.152*"coronavirus" + -0.148*"remain" + -0.148*"cupertino" + -0.148*"apartment" + -0.145*"county" + -0.133*"gilroy" + -0.133*"farm" + 0.133*"refuse"
2020-06-03 03:51:03,052 : INFO : topic #3(1.827): -0.172*"killing" + -0.172*"spark" + -0.170*"death" + -0.151*"county" + -0.146*"officer" + -0.144*"record" + -0.139*"washington" + -0.128*"tuesday" + -0.126*"cling" + -0.126*"daughter"
2020-06-03 03:51:03,052 : INFO : topic #4(1.781): -0.156*"white" + 0.129*"covid-19" + -0.129*"officer" + 0.126*"record" + 0.122*"santa" + 0.118*"people" + -0.115*"angeles" + -0.115*"insurrection" + -0.115*"acquittal" + -0.115*"act"
(59, 5)
2020-06-03 03:51:03,071 : INFO : using serial LSI version on this node
2020-06-03 03:51:03,071 : INFO : updating model with new documents
2020-06-03 03:51:03,077 : INFO : preparing a new chunk of documents
2020-06-03 03:51:03,078 : INFO : using 100 extra samples and 2 power iterations
2020-06-03 03:51:03,078 : INFO : 1st phase: constructing (458, 110) action matrix
2020-06-03 03:51:03,079 : INFO : orthonormalizing (458, 110) action matrix
2020-06-03 03:51:03,087 : INFO : 2nd phase: running dense svd on (110, 59) matrix
2020-06-03 03:51:03,088 : INFO : computing the final decomposition
2020-06-03 03:51:03,088 : INFO : keeping 10 factors (discarding 42.944% of energy spectrum)
2020-06-03 03:51:03,088 : INFO : processed documents up to #59
2020-06-03 03:51:03,089 : INFO : topic #0(2.120): 0.325*"city" + 0.172*"george" + 0.172*"floyd" + 0.164*"distinguish" + 0.164*"arrest" + 0.164*"city official" + 0.164*"thousand" + 0.164*"refuse" + 0.164*"ongoing protest" + 0.164*"honor"
2020-06-03 03:51:03,089 : INFO : topic #1(1.977): 0.212*"city" + 0.204*"widespread destruction" + 0.204*"clash" + 0.204*"destruction" + 0.204*"instituting" + 0.204*"instituting curfew" + 0.204*"looter" + 0.204*"weekend" + 0.204*"widespread" + 0.204*"cause"
2020-06-03 03:51:03,089 : INFO : topic #2(1.910): 0.157*"silicon" + 0.157*"valley" + 0.152*"coronavirus" + 0.148*"cupertino" + 0.148*"remain" + 0.148*"apartment" + 0.145*"county" + 0.133*"farm" + 0.133*"gilroy" + -0.133*"refuse"
2020-06-03 03:51:03,089 : INFO : topic #3(1.827): 0.172*"killing" + 0.172*"spark" + 0.170*"death" + 0.151*"county" + 0.146*"officer" + 0.144*"record" + 0.139*"washington" + 0.128*"tuesday" + 0.126*"cling" + 0.126*"daughter"
2020-06-03 03:51:03,089 : INFO : topic #4(1.781): 0.156*"white" + -0.129*"covid-19" + 0.129*"officer" + -0.126*"record" + -0.122*"santa" + -0.118*"people" + 0.115*"angeles" + 0.115*"insurrection" + 0.115*"acquittal" + 0.115*"act"
(59, 10)
"""




gc.collect()





# clustering
for n in [5, 10]:
    file_name0 = '/Users/lucui/NLP_projects/20200602_web_doc/data/processing-lsi-{0}.csv'.format(n)
    taxonomy_keywords = pd.read_csv(file_name0, sep='\t', engine='python')
    taxonomy_cols = []
    for i in range(n):
        taxonomy_cols.append(i)
    x = df.iloc[:, taxonomy_cols].values
    kmeans3 = KMeans(n_clusters=3)
    y_kmeans3 = kmeans3.fit_predict(x)
    print(y_kmeans3) # "predicted" document labels
    kmeans3.cluster_centers_
    # array([[ 0.11217066, -0.06784725,  0.13870174,  0.10235818, -0.00737711,
    #      0.05588331,  0.02487473, -0.03345771, -0.0524634 ,  0.04141877],
    #    [ 0.72133057, -0.38884976, -0.47761465, -0.22359023,  0.02215128,
    #     -0.00439351,  0.06941398,  0.0034246 ,  0.09901869,  0.08694823],
    #    [ 0.60177549,  0.73938785,  0.16444553, -0.09643104, -0.0569858 ,
    #     -0.11256348,  0.0137124 ,  0.03060841, -0.00754485, -0.0613114 ]])
    Error =[]
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i).fit(x)
        kmeans.fit(x)
        Error.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, 11), Error)
    plt.title('Elbow method')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()




path = "/Users/lucui/NLP_projects/20200602_web_doc/data/url_text_w_id.csv"
article_df = pd.read_csv(path, keep_default_na=False, sep='|', encoding="ISO-8859-1")

# predicted labels
label = [0,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,2,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
df_label = pd.DataFrame(label, columns=['label'])
concatenated = pd.concat([df_label, article_df], axis=1)
print(concatenated[['label','text']])

"""
>>> print(concatenated[['label','text']])
    label                                                                                                                                                                                                                                                                                                                                                                                                                                                       text
0   0      A veteran Bay Area real estate company has bought a big apartment complex in Cupertino that's located a mile away from Apple's vast spaceship campus, in a deal that points to a Silicon Valley market that remains robust despite the coronavirus. A purchase of Gardens of Fontainbleu Apartments in Cupertino suggests plenty of interest remains in choice Silicon Valley properties, despite grim economic effects from the coronavirus pandemic.   
1   0      Government-imposed business shutdowns helped to erase nearly 597,000 jobs in the Bay Area during March and April -- yet they also helped to wipe out vehicle trips and traffic jams in a big way, according to a new economic report released Tuesday.Â  With a fast-rising number of people working from home â or out of work entirely â people simply have been driving much less during a spring dominated by the personal and economic fallouâ¦
2   1      For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police. For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police.                                                                                                        
3   0      Sports Inside the Warriors Fast Break Warriors Fan Blog Giants Extra Inside the Aâs 49ers Hot Read Inside the Oakland Raiders College Hotline Tech & Business SiliconBeat Politics & Goâ¦                                                                                                                                                                                                                                                             
4   0      WASHINGTON (AP) â Undeterred by curfews, protesters streamed back into the nationâs streets Tuesday, hours after President Donald Trump pressed governors to put down the violence set ofâ¦                                                                                                                                                                                                                                                         
5   0      Santa Clara and Alameda counties each recorded a death from coronavirus, and Contra Costa County went over 1,500 confirmed COVID-19 cases during the pandemic with its biggest one-day spike in six weeks, health officials said Tuesday. Santa Clara County has recorded 143 deaths since the pandemic started, the most in the 10-county Bay Area, while Alameda County has recorded the most cases with more than 3,500.                              
6   0      Without knowing how the COVID-19 outbreak will continue to unfold, Gov. Cooper said, âplanning for a scaled-down convention with fewer people, social distancing and face coverings is a necessâ¦                                                                                                                                                                                                                                                     
7   0      How to contact the Bay Area News Group Main phone numbers and addresses The Mercury News: 408-920-5000 (voice) 408-288-8060 (fax) 4 N. Second Street Suite 800 San Jose, CA 95113 East Bay Times: 925â¦                                                                                                                                                                                                                                                 
8   0      A veteran and savvy Bay Area development company has bought the big Uesugi Farms site in Gilroy in a deal that points to ongoing interest in Silicon Valley commercial real estate amid the coronavirus outbreak. The purchase of 69 acres of land in Gilroy near the San Benito County line rescues a storied farm site that had operated as a family-run agriculture enterprise for decades from the early stages of â¦                               
9   0      The Insurrection Act hasn't been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King. The Insurrection Act hasnât been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King.                                                                                            
10  2      City officials addressed the ongoing protests in honor of George Floyd, continuing to distinguish between the thousands who peacefully marched Monday afternoon from a group arrested after refusing â¦                                                                                                                                                                                                                                                 
11  0      A veteran Bay Area real estate company has bought a big apartment complex in Cupertino that's located a mile away from Apple's vast spaceship campus, in a deal that points to a Silicon Valley market that remains robust despite the coronavirus. A purchase of Gardens of Fontainbleu Apartments in Cupertino suggests plenty of interest remains in choice Silicon Valley properties, despite grim economic effects from the coronavirus pandemic.   
12  0      The Insurrection Act hasn't been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King. The Insurrection Act hasnât been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King.                                                                                            
13  0      A veteran and savvy Bay Area development company has bought the big Uesugi Farms site in Gilroy in a deal that points to ongoing interest in Silicon Valley commercial real estate amid the coronavirus outbreak. The purchase of 69 acres of land in Gilroy near the San Benito County line rescues a storied farm site that had operated as a family-run agriculture enterprise for decades from the early stages of â¦                               
14  0      Santa Clara and Alameda counties each recorded a death from coronavirus, and Contra Costa County went over 1,500 confirmed COVID-19 cases during the pandemic with its biggest one-day spike in six weeks, health officials said Tuesday. Santa Clara County has recorded 143 deaths since the pandemic started, the most in the 10-county Bay Area, while Alameda County has recorded the most cases with more than 3,500.                              
15  0      Some health experts warn tear gas could âincrease risk for COVID-19 by making the respiratory tract more susceptible to infection, exacerbating existing inflammation, and inducing coughing.&#â¦                                                                                                                                                                                                                                                     
16  0      San Mateo County and Palo Alto on Tuesday joined other Bay Area jurisdictions in imposing curfews amid continuing unrest and looting sparked by the police killing of George Floyd in Minneapolis.                                                                                                                                                                                                                                                       
17  0      Without knowing how the COVID-19 outbreak will continue to unfold, Gov. Cooper said, âplanning for a scaled-down convention with fewer people, social distancing and face coverings is a necessâ¦                                                                                                                                                                                                                                                     
18  0      With her 6-year-old daughter Gianna clinging to her, Roxie Washington told reporters she wants all four officers involved in Floydâs death to pay for the killing, which has sparked protests aâ¦                                                                                                                                                                                                                                                     
19  0      *** The Pac-12 Hotline newsletter is published each Monday-Wednesday-Friday during the college sports season and twice-a-week in the summer. (Sign up here for a free subscription.) This edition, frâ¦                                                                                                                                                                                                                                                 
20  0      Latest sports news, commentary analysis, photos and videos about the 49ers, Raiders, Warriors, Giants, Athletics, Sharks, Earthquakes, Stanford Cardinal, Cal Bears and more Bay Area teams, from The Mercury News. bay area sports, bay area sports commentary, bay area sports news, NBA, NFL, MLB, NHL, 49ers, warriors, raiders, giants, athletics, sharks, earthquakes, stanford cardinal, cal bears, san jose spartans                             
21  1      For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police. For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police.                                                                                                        
22  0      Some health experts warn tear gas could âincrease risk for COVID-19 by making the respiratory tract more susceptible to infection, exacerbating existing inflammation, and inducing coughing.&#â¦                                                                                                                                                                                                                                                     
23  0      Bay Area News Group: Reprints of articles and photos Â  Articles: To license or republish Bay Area News Group articles and headlines in a book, documentary or other media, please contact our veâ¦                                                                                                                                                                                                                                                     
24  0      The Insurrection Act hasn't been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King. The Insurrection Act hasnât been invoked since 1992, during the riots in Los Angeles that followed the acquittal of four white police officers in the beating of Rodney King.                                                                                            
25  0      With her 6-year-old daughter Gianna clinging to her, Roxie Washington told reporters she wants all four officers involved in Floydâs death to pay for the killing, which has sparked protests aâ¦                                                                                                                                                                                                                                                     
26  0      San Mateo County and Palo Alto on Tuesday joined other Bay Area jurisdictions in imposing curfews amid continuing unrest and looting sparked by the police killing of George Floyd in Minneapolis.                                                                                                                                                                                                                                                       
27  1      For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police. For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police.                                                                                                        
28  2      City officials addressed the ongoing protests in honor of George Floyd, continuing to distinguish between the thousands who peacefully marched Monday afternoon from a group arrested after refusing â¦                                                                                                                                                                                                                                                 
29  0      San Mateo County and Palo Alto on Tuesday joined other Bay Area jurisdictions in imposing curfews amid continuing unrest and looting sparked by the police killing of George Floyd in Minneapolis.                                                                                                                                                                                                                                                       
30  0      Latest news from Atherton, Burlingame, East Palo Alto, Half Moon Bay, Menlo Park, Pacifica, Redwood City, San Bruno, San Mateo, South San Francisco and other San Mateo County cities                                                                                                                                                                                                                                                                    
31  0      Digital Access FAQ Thank you for being a reader of The Mercury News. Weâre proud to bring you the work of Northern Californiaâs largest news team as we continue a long tradition of aggrâ¦                                                                                                                                                                                                                                                         
32  2      City officials addressed the ongoing protests in honor of George Floyd, continuing to distinguish between the thousands who peacefully marched Monday afternoon from a group arrested after refusing â¦                                                                                                                                                                                                                                                 
33  0      Puzzles and Games Bridge Word Game Dear readers:Â We recently changed some of the online puzzles offered on this site. Not finding your favorite puzzle? Explore more puzzles here.                                                                                                                                                                                                                                                                      
34  2      City officials addressed the ongoing protests in honor of George Floyd, continuing to distinguish between the thousands who peacefully marched Monday afternoon from a group arrested after refusing â¦                                                                                                                                                                                                                                                 
35  0      Several high profile state parks that had been completely closed also have reopened in recent days, including Big Basin Redwoods and Castle Rock in the Santa Cruz Mountains.                                                                                                                                                                                                                                                                            
36  0      The Ghost Ship criminal retrial of master tenant Derick Almena was expected to begin in July, but because of coronavirus-related shelter-in-place, the court delayed the start. The Ghost Ship criminal retrial of master tenant Derick Almena was expected to begin in July, but because of coronavirus-related shelter-in-place, the court delayed the start.                                                                                          
37  0      Government-imposed business shutdowns helped to erase nearly 597,000 jobs in the Bay Area during March and April -- yet they also helped to wipe out vehicle trips and traffic jams in a big way, according to a new economic report released Tuesday.Â  With a fast-rising number of people working from home â or out of work entirely â people simply have been driving much less during a spring dominated by the personal and economic fallouâ¦
38  0      Other polling also found a lack of trust in law enforcement. A CBS News/YouGov poll showed a majority of Americans (57%) believe the police in most communities treat whites better than blacks, whilâ¦                                                                                                                                                                                                                                                 
39  0      WASHINGTON (AP) â Undeterred by curfews, protesters streamed back into the nationâs streets Tuesday, hours after President Donald Trump pressed governors to put down the violence set ofâ¦                                                                                                                                                                                                                                                         
40  0      Several high profile state parks that had been completely closed also have reopened in recent days, including Big Basin Redwoods and Castle Rock in the Santa Cruz Mountains.                                                                                                                                                                                                                                                                            
41  0      Several high profile state parks that had been completely closed also have reopened in recent days, including Big Basin Redwoods and Castle Rock in the Santa Cruz Mountains.                                                                                                                                                                                                                                                                            
42  0      *** The Pac-12 Hotline newsletter is published each Monday-Wednesday-Friday during the college sports season and twice-a-week in the summer. (Sign up here for a free subscription.) This edition, frâ¦                                                                                                                                                                                                                                                 
43  0      Without knowing how the COVID-19 outbreak will continue to unfold, Gov. Cooper said, âplanning for a scaled-down convention with fewer people, social distancing and face coverings is a necessâ¦                                                                                                                                                                                                                                                     
44  1      For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police. For some cities, instituting curfews may have helped, following a weekend in which looters tore through cities, causing widespread destruction and clashing with police.                                                                                                        
45  0      Latest sports news, commentary analysis, photos and videos about the 49ers, Raiders, Warriors, Giants, Athletics, Sharks, Earthquakes, Stanford Cardinal, Cal Bears and more Bay Area teams, from The Mercury News. bay area sports, bay area sports commentary, bay area sports news, NBA, NFL, MLB, NHL, 49ers, warriors, raiders, giants, athletics, sharks, earthquakes, stanford cardinal, cal bears, san jose spartans                             
46  0      A veteran Bay Area real estate company has bought a big apartment complex in Cupertino that's located a mile away from Apple's vast spaceship campus, in a deal that points to a Silicon Valley market that remains robust despite the coronavirus. A purchase of Gardens of Fontainbleu Apartments in Cupertino suggests plenty of interest remains in choice Silicon Valley properties, despite grim economic effects from the coronavirus pandemic.   
47  0      With her 6-year-old daughter Gianna clinging to her, Roxie Washington told reporters she wants all four officers involved in Floydâs death to pay for the killing, which has sparked protests aâ¦                                                                                                                                                                                                                                                     
48  0      A veteran and savvy Bay Area development company has bought the big Uesugi Farms site in Gilroy in a deal that points to ongoing interest in Silicon Valley commercial real estate amid the coronavirus outbreak. The purchase of 69 acres of land in Gilroy near the San Benito County line rescues a storied farm site that had operated as a family-run agriculture enterprise for decades from the early stages of â¦                               
49  0      WASHINGTON (AP) â Undeterred by curfews, protesters streamed back into the nationâs streets Tuesday, hours after President Donald Trump pressed governors to put down the violence set ofâ¦                                                                                                                                                                                                                                                         
50  0      Santa Clara and Alameda counties each recorded a death from coronavirus, and Contra Costa County went over 1,500 confirmed COVID-19 cases during the pandemic with its biggest one-day spike in six weeks, health officials said Tuesday. Santa Clara County has recorded 143 deaths since the pandemic started, the most in the 10-county Bay Area, while Alameda County has recorded the most cases with more than 3,500.                              
51  0      *** The Pac-12 Hotline newsletter is published each Monday-Wednesday-Friday during the college sports season and twice-a-week in the summer. (Sign up here for a free subscription.) This edition, frâ¦                                                                                                                                                                                                                                                 
52  0      Ad Placement & Newspaper Advertising Please visit bayareanewsgroup.com to browse the full portfolio of our advertising solutions. Classifieds Advertising To place a classified ad, please visit â¦                                                                                                                                                                                                                                                     
53  0      san jose sharks, sharks hockey, sharks news, san jose sharks news, sharks players, sharks coach, sharks standings, sharks analysis, sharks roster, sharks schedule, sharks latest news, sharks game, sharks score, sharks photos                                                                                                                                                                                                                         
54  0      Some health experts warn tear gas could âincrease risk for COVID-19 by making the respiratory tract more susceptible to infection, exacerbating existing inflammation, and inducing coughing.&#â¦                                                                                                                                                                                                                                                     
55  0      crime, traffic, Golden State Warriors, San Francisco 49ers, San Francisco Giants, Oakland Raiders, Oakland Athletics, San Jose Sharks, San Jose Earthquakes, Silicon Valley                                                                                                                                                                                                                                                                              
56  0      Government-imposed business shutdowns helped to erase nearly 597,000 jobs in the Bay Area during March and April -- yet they also helped to wipe out vehicle trips and traffic jams in a big way, according to a new economic report released Tuesday.Â  With a fast-rising number of people working from home â or out of work entirely â people simply have been driving much less during a spring dominated by the personal and economic fallouâ¦
57  0      Other polling also found a lack of trust in law enforcement. A CBS News/YouGov poll showed a majority of Americans (57%) believe the police in most communities treat whites better than blacks, whilâ¦                                                                                                                                                                                                                                                 
58  0      Other polling also found a lack of trust in law enforcement. A CBS News/YouGov poll showed a majority of Americans (57%) believe the police in most communities treat whites better than blacks, whilâ
"""

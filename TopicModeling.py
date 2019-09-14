import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords # To remove stopwords
# To create corpus and dictionary for the LDA model
import gensim
from gensim import corpora 
from gensim.models import LdaModel # To use the LDA model
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import operator
import pandas as pd
import gzip

# one-review-per-line in json 
#Code refrence -  http://jmcauley.ucsd.edu/data/amazon/ 

# Step 1: Extract Data

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF(r'C:\Users\AKhushaiani\Desktop\NLP\reviews_Electronics_5.json.gz')
df.head(10)
df.columns

# Step 2: Tokenizer, remove stopwords, lower case all text

#Regular expression tokenizer
tokenizer = RegexpTokenizer(r'\w+')
doc_1 = df.reviewText[0]

# Using one of review
tokens = tokenizer.tokenize(doc_1.lower())

print('{} characters in string vs {} words in a list'.format(len(doc_1),                                                             len(tokens)))
print(tokens[:10])

nltk_stpwd = stopwords.words('english')

print(len(set(nltk_stpwd)))
print(nltk_stpwd[:10])

stopped_tokens = [token for token in tokens if not token in nltk_stpwd]
print(stopped_tokens[:10])

# Step 3: Stemming using Snowball stemmer

sb_stemmer = SnowballStemmer('english')
stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]
print(stemmed_tokens)

num_reviews = df.shape[0]

doc_set = [df.reviewText[i] for i in range(num_reviews)]

texts = []

for doc in doc_set:
    tokens = tokenizer.tokenize(doc.lower())
    stopped_tokens = [token for token in tokens if not token in nltk_stpwd]
    stemmed_tokens = [sb_stemmer.stem(token) for token in stopped_tokens]
    texts.append(stemmed_tokens)# Adds tokens to new list "texts"
    
print(texts[1])

#Step 4: Create a dictonary using corpora
texts_dict = corpora.Dictionary(texts)
texts_dict.save('elec_review.dict') 
print(texts_dict)

#Assess the mapping between words and their ids I use the token2id method:
print("IDs 1 through 10: {}".format(sorted(texts_dict.token2id.items(), key=operator.itemgetter(1), reverse = False)[:10]))

#Here I assess how many reviews have word complaint in it
complaints = df.reviewText.str.contains("complaint").value_counts()
ax = complaints.plot.bar(rot=0)

"""
Attempting to see what happens if I ignore tokens that appear in less 
than 30 documents or more than 20% documents.
"""

texts_dict.filter_extremes(no_below=20, no_above=0.10) 
print(sorted(texts_dict.token2id.items(), key=operator.itemgetter(1), reverse = False)[:10])

# Step 5: Converting the dictionary to bag of words calling it corpus here
corpus = [texts_dict.doc2bow(text) for text in texts]
len(corpus)

#Save a corpus to disk in the sparse coordinate Matrix Market format in a serialized format instead of random
gensim.corpora.MmCorpus.serialize('amzn_elec_review.mm', corpus)

"""

The number of topics is random, using the https://www.amazon.com/exec/obidos/tg/browse/-/172282 to determine the number of topics
1. Computer - Accesories
2. TV & Video
3. Cell Phones & Accesories
4. Photlography & Videography
5. Home Audio
6. Amazon devices
7. Headphones
8. Office Electronics
9. Office supplies
10. Smart Home
11. Musical Instruments
12. Video Games

"""
 
#Step 6: Fit LDA model
lda_model = gensim.models.LdaModel(corpus,alpha='auto', num_topics=5,id2word=texts_dict, passes=20)

#Choosing the number of topics based on various categories of electronics on Amazon
lda_model.show_topics(num_topics=5,num_words=5)

raw_query = 'portable speaker'

query_words = raw_query.split()
query = []
for word in query_words:
    # ad-hoc reuse steps from above
    q_tokens = tokenizer.tokenize(word.lower())
    q_stopped_tokens = [word for word in q_tokens if not word in nltk_stpwd]
    q_stemmed_tokens = [sb_stemmer.stem(word) for word in q_stopped_tokens]
    query.append(q_stemmed_tokens[0])
    
print(query)

# Words in query will be converted to ids and frequencies  
id2word = gensim.corpora.Dictionary()
_ = id2word.merge_with(texts_dict) # garbage

# Convert this document into (word, frequency) pairs
query = id2word.doc2bow(query)
print(query)

#Create a sorted list
sorted_list = list(sorted(lda_model[query], key=lambda x: x[1]))
sorted_list

#Assessing least related topics
lda_model.print_topic(a[0][0]) #least related

#Assessing most related topics
lda_model.print_topic(a[-1][0]) #most related





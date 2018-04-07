from operator import itemgetter
from datetime import datetime, timedelta
import itertools
import math
import re
import os

import requests
from dateutil import parser
from bs4 import BeautifulSoup
from django.utils.timezone import utc
import nltk
import pytz
from stemming.porter2 import stem
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag import StanfordNERTagger


def createArticleByContent(title, subtitle, body, reference):
    keywords = extracKeywords(title + ', ' + subtitle + ', ' + body.split(".")[0])
    date = datetime.utcnow().replace(tzinfo=utc)
    article = createArticleObject("User_"+reference,title, subtitle, body, date, keywords, None, "UserArticle", "UserContent", reference)
    return article


def createArticleObject(ID, title, subtitle, body, date, keywords, url, type, source, reference=None):
    # print(date)
    time_threshold = date - timedelta(hours=24)
    list_of_article = Article.objects.filter(DateTime__gte=time_threshold, DateTime__lte=date).only("Stemming_Title",
                                                                                                    "Stemming_Content")
    stem_title = stemming(parseContent(title + ' ' + subtitle + ' ' + body.split('.')[0]))
    stem_content = stemming(parseContent(body))
    profile = getArticleProfile(stem_title, stem_content, list_of_article)

    sent_detection,st,tokenizer = stanfordNERInit()
    keyword = keywords.split(',')
    entities = stanfordNER(title + ' ' + subtitle + ' ' + body.split('.')[0],sent_detection,st,tokenizer)
    print("keywords:",keywords)
    print("entities:",entities)
    pairdKeyword_str = pairKeywordsEntities(keyword,entities,tfidf)
    print("paird keywords with ner",pairdKeyword_str)
    pairdKeyword_str2 = pairKeywords(keywords, profile)
    print("paird keywords without ner",pairdKeyword_str2)
    article = Article(ID=ID, Headline=title, SubHeadline=subtitle,
                      Content=body, Url=url,
                      DateTime=date, Keywords=keywords,
                      Stream_Keywords=pairdKeyword_str,
                      Stemming_Title=stem_title,
                      Stemming_Content=stem_content,
                      Profile=profile,
                      Type=type,
                      Source=source,
                      Reference=reference,
                      NumberTweets=0
    )
    return article

def pairKeywordsEntities(keywords, entities, profile):
    final = []
    toPair = []
    toPair_phrase = []
    # print("keywords:", keyword_string)
    entities_string = [x for x,y in entities]
    print(entities_string)
    entities_dict = dict(entities)
    keywords = list(set(keywords+entities_string))
    for Keyword in keywords:
        if ' ' in Keyword.strip():
            toPair_phrase = Keyword.strip().split(' ')
            toPair_phrase = sorted(toPair_phrase, key=str.lower)
            for combination in itertools.combinations(toPair_phrase, 2):
                final.append(' '.join(combination))
        else:
            toPair.append(Keyword.strip())

    toPair = sorted(list(set(toPair)), key=str.lower)
    if len(toPair) > 1:
        for combination in itertools.combinations(toPair, 2):
            final.append(' '.join(combination))

    # print(final)
    final = list(set(final))
    #print("uniq final:", final)
    #print(profile)
    if profile:
        try:
            final_scored = {}
            for pair in final:
                #print("pair: ", pair)
                score_one = 0.001
                score_two = 0.001
                score_three = 1
                score_four = 1
                term1 = pair.split()[0]
                term2 = pair.split()[1]
                try:
                    score_one = profile[stemming(term1)]
                    score_two = profile[stemming(term2)]
                except:
                    # print(term1,term2)
                    pass

                for key in entities_dict.keys():
                    if term1 in key or term1 == key:
                        if entities_dict[key] == 'PERSON' or entities_dict[key] == 'ORGANIZATION':
                            score_three = 3
                            break
                        elif entities_dict[key] == 'LOCATION':
                            score_three = 2
                            break
                    else:
                        score_three = 1
                for key in entities_dict.keys():
                    if term2 in key or term2 == key:
                        if entities_dict[key] == 'PERSON' or entities_dict[key] == 'ORGANIZATION':
                            score_four = 3
                            break
                        elif entities_dict[key] == 'LOCATION':
                            score_four = 2
                            break
                    else:
                        score_four = 1

                final_scored[pair] = score_one * score_two * score_three * score_four
                #print("score: ", final_scored[pair])
            final_sorted = sorted(final_scored.items(), key=itemgetter(1), reverse=True)
            #print("final_sorted:", final_sorted)
            ranked_pairs = []
            for pair in final_sorted:
                ranked_pairs.append(pair[0])
            return ','.join(ranked_pairs)
        except:
            return ','.join(final)
    else:
        return ','.join(final)



def stanfordNERInit():
    os.environ['CLASSPATH'] = 'C:/users/home/stanford-ner/stanford-ner-2017-06-09/stanford-ner.jar:C:/users/home/stanford-ner/stanford-ner-2017-06-09/lib/*:C:/users/home/stanford-postagger-full-2017-06-09/stanford-postagger.jar'
    os.environ['STANFORD_MODELS'] = 'C:/users/home/stanford-ner/stanford-ner-2017-06-09/classifiers/'
    sent_detection = nltk.data.load('tokenizers/punkt/english.pickle')
    st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
    tokenizer = StanfordTokenizer()

    return sent_detection,st,tokenizer


def stanfordNER(Headline,sent_detection,st,tokenizer):

    sentences = sent_detection.tokenize(Headline)
    sentences = [x for x in sentences if len(x)>5]
    # print(sentences)

    alltags = []
    for sentence in sentences:
        sentence = sentence.replace("’","")
        tags = st.tag(tokenizer.tokenize(sentence.replace("‘","")))
        tags = geContiChunks(tags)
        alltags.extend(tags)
        # print(sentence)
        # print(tags)

    return [(x.lower(),y) for x,y in alltags]


def geContiChunks(tags):
    contiChunks = []
    currentChunks = []

    for token,label in tags:
        if label != 'O':
            currentChunks.append((token,label))
        else:
            if currentChunks:
                temp = (' '.join([x for x,y in currentChunks]),currentChunks[0][1])
                contiChunks.append(temp)
                currentChunks =[]

    if currentChunks:
        temp = (' '.join([x for x,y in currentChunks]),currentChunks[0][1])
        contiChunks.append(temp)
    return contiChunks

# def pairKeywords(keyword_string):
# final = []
#     toPair = []
#     toPair_phrase = []
#     Keywords = keyword_string.split(',')
#     for Keyword in Keywords:
#         if ' ' in Keyword.strip():
#             toPair_phrase = Keyword.strip().split(' ')
#             toPair_phrase = sorted(toPair_phrase, key=str.lower)
#             for combination in itertools.combinations(toPair_phrase, 2):
#                 final.append(' '.join(combination))
#         else:
#             toPair.append(Keyword.strip())
#
#     toPair = sorted(toPair, key=str.lower)
#     if len(toPair) > 1:
#         for combination in itertools.combinations(toPair, 2):
#             final.append(' '.join(combination))
#
#     return ','.join(final)

def pairKeywords(keyword_string, profile):
    final = []
    toPair = []
    toPair_phrase = []
    # print("keywords:", keyword_string)
    Keywords = keyword_string.split(',')
    for Keyword in Keywords:
        if ' ' in Keyword.strip():
            toPair_phrase = Keyword.strip().split(' ')
            toPair_phrase = sorted(toPair_phrase, key=str.lower)
            for combination in itertools.combinations(toPair_phrase, 2):
                final.append(' '.join(combination))
        else:
            toPair.append(Keyword.strip())

    toPair = sorted(toPair, key=str.lower)
    if len(toPair) > 1:
        for combination in itertools.combinations(toPair, 2):
            final.append(' '.join(combination))

    # print(final)
    final = list(set(final))
    #print("uniq final:", final)
    #print(profile)
    if profile:
        try:
            final_scored = {}
            for pair in final:
                #print("pair: ", pair)
                score_one = 0
                score_two = 0
                try:
                    #print("stem:", stemming(pair.split()[0]))
                    #print("stem:", stemming(pair.split()[1]))
                    score_one = profile[stemming(pair.split()[0])]
                    score_two = profile[stemming(pair.split()[1])]
                except:
                    # print(profile)
                    # print("except error on pair:", pair)
                    pass
                final_scored[pair] = score_one * score_two
                #print("score: ", final_scored[pair])
            final_sorted = sorted(final_scored.items(), key=itemgetter(1), reverse=True)
            #print("final_sorted:", final_sorted)
            ranked_pairs = []
            for pair in final_sorted:
                ranked_pairs.append(pair[0])
            return ','.join(ranked_pairs)
        except:
            return ','.join(final)
    else:
        return ','.join(final)


def stemming(sentence):
    lmtzr = WordNetLemmatizer()
    words = sentence.split(' ')
    result = ''
    for word in words:
        result += stem(lmtzr.lemmatize(word)) + ' '
    return result.strip()


def parseContent(content):
    content = content.lower()
    content = re.sub('[:;>?<=*+()./,\-#!&"$%\{˜|\}\'\[ˆ_\\@\]1234567890’‘]', ' ', content)
    content = removeStopWords(content)

    return content


def getArticleProfile(Stemming_Title, Stemming_Content, list_of_articles):
    if list_of_articles.exists():
        arti_content_list = []
        arti_content_list.append(Stemming_Title + Stemming_Content)
        for article in list_of_articles.iterator():
            arti_content_list.append(article.Stemming_Title + article.Stemming_Content)

        tfidf_result = tfidf(Stemming_Title.split(' '), Stemming_Title.split(' ') + Stemming_Content.split(' '),
                             arti_content_list)
    else:
        tfidf_result = tfidf_2(Stemming_Title.split(' '), Stemming_Title.split(' ') + Stemming_Content.split(' '))

    tfidf_result = profile_normal(tfidf_result)

    return tfidf_result


def tfidf(words, article, articlelist):
    tf_ = tf(words, article)
    idf_ = idf(words, articlelist)
    temp = [round(x * y, 4) for x, y in zip(tf_, idf_)]
    tf_idf = dict(zip(words, temp))
    return tf_idf


def tfidf_2(words, article):
    tf_ = tf(words, article)
    temp = [round(x, 4) for x in tf_]
    tf_idf = dict(zip(words, temp))
    return tf_idf


def profile_normal(tf_idf):
    temp = math.sqrt(math.fsum(x * x for x in tf_idf.values()))
    if 0 != temp:
        for key in tf_idf.keys():
            tf_idf[key] = round(tf_idf[key] / temp, 4)
        return tf_idf
    else:
        for key in tf_idf.keys():
            tf_idf[key] = 0
        return tf_idf


def tf(words, article):
    wordsfreq = [sum(p == word for p in article) for word in words]
    maxfreq = max(wordsfreq)
    wordsfreq = [math.log((x / maxfreq) + 1) for x in wordsfreq]
    return wordsfreq


def n_containing(word, articlelist):
    t = 0
    for article in articlelist:
        if word in article:
            t += 1
    if t is 0:
        return 1
    return t


def idf(words, articlelist):
    t = [math.log(len(articlelist) / (n_containing(word, articlelist))) for word in words]
    return t

def extracKeywords(Headlines):
    stop_words = load_stopwords()
    result_keyword = []
    tokens = nltk.word_tokenize(Headlines)
    # clean_tokens = fix_tokens(tokens, stop_words)
    pos_tokens = nltk.pos_tag(tokens)

    nouns = []

    grammar = "NP: {<NNP><NNP>(<NNP>*)|<NNP><CC><NNP>}"
    cp = nltk.RegexpParser(grammar)
    entities = cp.parse(pos_tokens)

    for pos_token in entities:
        pos_token = str(pos_token)
        # print pos_token
        if 'NN' in pos_token or 'NNP' in pos_token or 'NNS' in pos_token or 'NP' in pos_token:
            noun = pos_token.split(",")[0].replace("(", "").replace(")", "").replace("'", "").replace("NNP",
                                                                                                      "").replace("NNS",
                                                                                                                  "").replace(
                "NN", "").replace("NP", "").replace("/", "").lower()

            if noun not in stop_words:
                nouns.append(pos_token.lower())

    nouns_freq = nltk.FreqDist(nouns)
    top_keys = nouns_freq.keys()
    top_values = nouns_freq.values()

    selected_nouns = []
    noun_phrases = []
    frequent_nouns = []
    nnp_nouns = []
    nnp_and_freq_nouns = []
    other_nouns = []
    for keyword in top_keys:
        top_noun = keyword.split(",")[0].replace("(", "").replace(")", "").replace("'", "")
        if '/' in top_noun:  # this is a Noun Phrase (Named Entity) "NP United/NNP Nations/NNP"
            top_noun_entity = top_noun.split(" ")
            np_words = ''
            for word in top_noun_entity[1:]:
                np_words += word.split("/")[0] + " "
            noun_phrases.append(np_words)
        else:  # this is a simple (noun, tag) pair
            top_noun_tag = keyword.split(",")[1].replace("(", "").replace(")", "").replace("'", "")
            #select high freqs proper nouns
            if nouns_freq[keyword] >= 2 and 'nnp' in top_noun_tag:
                nnp_and_freq_nouns.append(top_noun)
            else:
                if nouns_freq[keyword] >= 2:
                    frequent_nouns.append(top_noun)
                #select NNP nouns
                else:
                    if 'nnp' in top_noun_tag:
                        nnp_nouns.append(top_noun)
                    else:
                        other_nouns.append(top_noun)

    for noun in nnp_and_freq_nouns:
        if "." not in noun:
            selected_nouns.append(noun)
    for noun in nnp_nouns:
        if "." not in noun:
            selected_nouns.append(noun)
    for noun in frequent_nouns:
        selected_nouns.append(noun)
    for noun in other_nouns:
        selected_nouns.append(noun)

        # print selected_nouns

    #
    # for i in range(0, min(2, len(selected_nouns))):
    # if len(selected_nouns) >= 2:
    #
    #         result_keyword.append(selected_nouns[i])

    for phrase in noun_phrases:
        #need to replace & for tokens like h&m, s&p
        phrase = phrase.replace(" & ", "&")
        #replace the 7 coordinating conjunctions (CC) with space
        phrase = phrase.replace(" and ", " ")
        phrase = phrase.replace(" or ", " ")
        phrase = phrase.replace(" for ", " ")
        phrase = phrase.replace(" so ", " ")
        phrase = phrase.replace(" nor ", " ")
        phrase = phrase.replace(" yet ", " ")
        phrase = phrase.replace(" but ", " ")
        phrase = phrase.replace(" @ ", " @")
        if "." not in phrase:
            result_keyword.append(phrase.strip())

    for select in selected_nouns:
        label = True
        for phrase in noun_phrases:
            if select in phrase:
                label = False
                break
        if label:
            result_keyword.append(select)
        result_keyword = list(set([word for word in result_keyword if len(word) > 1]))
        if len(result_keyword) >= 5:
            break

    return ", ".join(result_keyword)


def load_stopwords():
    stop_words = nltk.corpus.stopwords.words('english')
    # custom stop words for avoiding retrieving too much spam from Twitter
    stop_words.append("video")
    stop_words.append("videos")
    stop_words.append("anyone")
    stop_words.append("today")
    stop_words.append("new")
    stop_words.append("former")
    stop_words.append("cent")
    stop_words.append("image")
    stop_words.append("images")
    stop_words.append("want")
    stop_words.append("yes")
    stop_words.append("no")
    stop_words.append("on")
    stop_words.append("dont")
    stop_words.append(".")
    stop_words.append("inside")
    stop_words.append("first")
    stop_words.append("immense")
    stop_words.append("simple")
    stop_words.append("finds")
    stop_words.append("best")
    stop_words.append("large")
    stop_words.append("huge")
    stop_words.append("regardless")
    stop_words.append("latest")
    stop_words.append("proud")
    stop_words.append("as")
    stop_words.append("although")
    stop_words.append("...")
    stop_words.append("bbc")
    stop_words.append("news")
    stop_words.append("either")
    stop_words.extend(['this', 'that', 'the', 'might', 'have', 'been', 'from',
                       'but', 'they', 'will', 'has', 'having', 'had', 'how', 'went'
                                                                             'were', 'why', 'and', 'still', 'his',
                       'her',
                       'was', 'its', 'per', 'cent',
                       'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
                       'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can',
                       'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every',
                       'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his',
                       'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let',
                       'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
                       'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said',
                       'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their',
                       'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
                       'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
                       'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your', 've', 're', 'rt'])

    #turn list into set for faster search
    stop_words = set(stop_words)
    return stop_words


def removeStopWords(tweet):
    stopword = load_stopwords()
    stopword = list(stopword)
    split = str.split(tweet, ' ')
    for s in split:
        if len(s) < 2 or s in stopword:
            split[split.index(s)] = ''
    tweet = ' '.join(split)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = tweet.strip()
    return tweet

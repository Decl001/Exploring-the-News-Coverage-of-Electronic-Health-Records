'''
This module contains all methods for annotating the
article datasets. It also calls the methods from 
TAGME_Filtering to filter the data. It creates new
datasets from the ones that are passed to it which 
have the annotated data included in them.
'''

import pandas as pd
import numpy as np
import json
import requests
import re
import TAGME_Filtering
import ProjectBaseMethods as pbm
from matplotlib import pyplot as plt

def getTAGMEannotations(article):
    #Configure settings for tagging
    TAGME_TOKEN = 'cb6ce53e-70af-4203-9c5b-423525722091-843339462'
    TAGME_ENDPOINT = 'https://tagme.d4science.org/tagme/tag'
    configs = {
        'lang':'en',
        'gcube-token':TAGME_TOKEN,
        'include_categories':'true',
        'epsilon':0.1,
        'long_text':10,
        'text' : article
    }
    try:
        result = requests.post(TAGME_ENDPOINT, configs)
        result = json.loads(result.text)
        return result['annotations']
    except Exception as e:
        print(e)
        return None

def addFilteredOrgsToDict(filteredResults, resultDict, yearPos):
    for result in filteredResults:
        if result in resultDict:
            resultDict[result][yearPos] +=1
        else:
            resultDict[result] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            resultDict[result][yearPos] +=1
    return resultDict

def annotate_data(project):
    for ds in project.datasets:
        df = []
        try:
            df = pd.read_csv(ds)
        except :
            try:
                df = pd.read_csv(ds,encoding='cp1252')
            except:
                raise Exception('Could not read datasets')
        fullAnnotations = []
        yearPos = 0
        resultDict = dict()
        for i,article in enumerate(df.article_body):
            if i > 0 and pbm.isNewHalfYear(df.date[i], df.date[i-1]) is not None:
                if pbm.isNewHalfYear(df.date[i], df.date[i-1]):
                    yearPos += 1
            if type(article) is str:
                annotations = getTAGMEannotations(article)
                if annotations is not None:
                    try:
                        filteredResults = TAGME_Filtering.filterByProbability(annotations)
                        relFilteredResults = TAGME_Filtering.filterByRelatedness(filteredResults)
                        resultDict = addFilteredOrgsToDict(relFilteredResults, resultDict, yearPos)
                        indexes_to_pop = []
                        for j,result in enumerate(list(filteredResults)):
                            if result['title'] not in relFilteredResults:
                                indexes_to_pop.append(j - len(indexes_to_pop))#account for index changes
                        for idx in indexes_to_pop:
                            filteredResults.pop(idx)
                        fullAnnotations.append(filteredResults)
                    except:
                        pass
        #save the annotated datset
        df['TAGME_Annotations'] = pd.Series(fullAnnotations)
        while '/' in ds:
            pre,sep,ds = ds.partition('/')
        df.to_csv(project.base_dir + '/annotated-data/' + ds + '_TAGME.csv')
        #plot the top 10 from each dataset
        tickLabels = pbm.generate_ticklabels(df)
        bins = np.arange(len(tickLabels))
        maxKeys = ['','','','','','','','','','']
        maxCounts = [0,0,0,0,0,0,0,0,0,0]
        for key in resultDict:
            for i,pos in enumerate(maxCounts):
                if sum(resultDict[key]) > pos:
                    maxKeys.insert(i,key)
                    maxCounts.insert(i,sum(resultDict[key]))
                    maxKeys.pop()
                    maxCounts.pop()
                    break
        for key in maxKeys:
            plt.bar(np.arange(len(tickLabels)),resultDict[key], label=key)
            plt.xticks(bins,tickLabels,rotation=45)
        plt.rcParams['figure.figsize'] = (25.0, 10.0)
        plt.legend(loc='upper right')
        plt.savefig(project.base_dir + '/tagme-counts/' + ds + '-top10.png')
        plt.clf()

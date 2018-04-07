import pandas as pd
import requests
import json

TAGME_TOKEN = 'cb6ce53e-70af-4203-9c5b-423525722091-843339462'
RELATEDNESS_ENDPOINT = 'https://tagme.d4science.org/tagme/rel'
rel_configs = {
    'gcube-token' : TAGME_TOKEN,
    'lang' : 'en',
}

def filterByProbability(retrivedEntities):
    '''
    Filters out the results that have below 0.1
    link_probability
    '''
    valid = []
    for spot in retrivedEntities:
        if spot['link_probability'] > 0.1:
            valid.append(spot)
    return valid

def filterByRelatedness(retrievedEntities):
    '''
    Filters out irrelevant entities by determing
    how many entities within the results they are 
    related to. 
    '''
    threshold = len(retrievedEntities)/4
    id_dict = generateIdDict(retrievedEntities)
    id_list = generateIdCouples(id_dict)
    #the endpoint can only handle 100 couples at a time
    rel_results = []
    first_pos = 0
    second_pos = 0
    if len(id_list) > 100:
        second_pos = 100
    else:
        second_pos = len(id_list)
    while(True):
        rel_configs['id'] = id_list[first_pos:second_pos]
        result = requests.get(RELATEDNESS_ENDPOINT, rel_configs)
        result = json.loads(result.text)
        rel_results += result['result']
        first_pos = second_pos
        if second_pos != len(id_list):
            second_pos += 100
            if second_pos > len(id_list):
                second_pos = len(id_list)
        else:
            break

    for result in rel_results:
        if result['rel'] > 0:
            coupleStr = result['couple']
            idPair = coupleStr.split(' ')
            id_dict[int(idPair[0])]['count'] += 1
            id_dict[int(idPair[1])]['count'] += 1
    
    filtered_results = []
    for key in id_dict:
        if id_dict[key]['count'] >= threshold:
            filtered_results.append(id_dict[key]['title'])
    
    return filtered_results

def generateIdDict(retrievedEntities):
    id_dict = {}
    for annot in retrievedEntities:
        id_dict[annot['id']] = {}
        id_dict[annot['id']]['title'] = annot['title']
        id_dict[annot['id']]['count'] = 0

    return id_dict  

def generateIdCouples(id_dict):
    id_list = []
    for i,entry in enumerate(id_dict):
        for j,otherEntry in enumerate(id_dict):
            if j > i:
                thisCouple = str(entry) + ' ' + str(otherEntry)
                id_list.append(thisCouple)
    return id_list
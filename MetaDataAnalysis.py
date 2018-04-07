import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ProjectBaseMethods as pbm

def plot_article_counts(project):
    datasets = project.datasets
    article_counts = []
    for ds in datasets:
        try:
            this_df = pd.read_csv(ds)
        except:
            try:
                this_df = pd.read_csv(ds,encoding='cp1252')
            except Exception:
                raise Exception('Unable to open dataset')#this is just to change the message so the 
                                                        #app text is more informative
        time_pos  = 0
        this_counts =[]
        for i,date in enumerate(this_df.date):
            if i != 0 and pbm.isNewHalfYear(date, this_df.date[i-1]):
                    time_pos += 1
            this_counts.append(time_pos)
        article_counts.append(this_counts)
    bins=range(min(article_counts[0]),max(article_counts[0])+1)
    plt.hist(article_counts,bins=bins, alpha=0.7)
    plt.xticks(bins,pbm.generate_ticklabels(pd.read_csv(datasets[0])),rotation=45)
    plt.savefig(project.base_dir + '/graphs/basic-counts/article_counts.png')
    plt.clf()

def groupArticlesByDate(articles, dataFrame):
    groupedByDate = []
    thisGroup = []
    for i,article in enumerate(articles):
        if i > 0:
            try:
                if pbm.isNewHalfYear(dataFrame.date[i], dataFrame.date[i-1]):
                    groupedByDate.append(thisGroup)
                    thisGroup = [str(article)]
                else:
                    thisGroup.append(str(article))
            except:
                pass
    return groupedByDate

def count_subjects(project):
    dfs = []
    df_subjects = []
    for ds in project.datasets:
        try:
            dfs.append(pd.read_csv(ds))
        except:
            try:
                dfs.append(pd.read_csv(ds,encoding='cp1252'))
            except:
                raise Exception('Could not read dataset')
    
    df_subjects.append(dfs[0].keywords)
    for i,df in enumerate(dfs):
        if i > 0:
            df_subjects.append([])
            for j,url in enumerate(df.url):
                for k,o_df in enumerate(dfs):
                    if k >= i:
                        break
                    if url in list(o_df.url):
                        df_subjects[i].append('duplicate')
                    else:
                        df_subjects[i].append(df.keywords[j])
    df_dateblocks = [groupArticlesByDate(this_subjects,dfs[i])
                        for i,this_subjects in enumerate(df_subjects)]
    fullDateBlocks = []
    for i,block_set in enumerate(df_dateblocks):
        for j,dateblock in enumerate(block_set):
            if i == 0:
                fullDateBlocks.append(dateblock)
            else:
                fullDateBlocks[j] += dateblock
    subjectsByBlock = dict()#dict will have subjects as the key and a list of occurences per block
    for i,dateBlock in enumerate(fullDateBlocks):
        for subjectString in dateBlock:
            if type(subjectString) is str:
                while ':' in subjectString:
                    typeStr,sep,subjectString = subjectString.partition(':')#definition of type (subject,type_of_material, etc)
                    subject,sep,subjectString = subjectString.partition(',')#what the subject is
                    if typeStr == 'subject':
                        if subject.lower() in subjectsByBlock:
                            subjectsByBlock[subject.lower()][i] += 1
                        else:
                            subjectsByBlock[subject.lower()] = [0 for j in range(len(fullDateBlocks))]
                            subjectsByBlock[subject.lower()][i] += 1
    tickLabels = pbm.generate_ticklabels(dfs[0])
    bins = np.arange(0,len(fullDateBlocks))
    count = 0
    for i,key in enumerate(subjectsByBlock):
        plt.bar(np.arange(0,20),subjectsByBlock[key], label=key)
        plt.xticks(bins,tickLabels,rotation=45)
        if i > 0 and i % 10 == 0:#show plots in groups of 10
            plt.rcParams['figure.figsize'] = (25.0, 10.0)
            plt.legend(loc='upper right')
            plt.savefig(project.base_dir + '/graphs/basic-counts/subjects' + str(count) + '.png')
            plt.clf()   
            count += 1
    #show the rest at the end
    plt.rcParams['figure.figsize'] = (25.0, 10.0)
    plt.legend(loc='upper right')
    plt.savefig(project.base_dir + '/graphs/basic-counts/subjects' + str(count) + '.png')
    plt.clf()              
    #pull out the max sum keys to get the 10
    #most discussed subjects
    dictForMax = dict(subjectsByBlock)#copy the dict
    for i in range(10):
        maxSum = 0
        maxKey = ""
        for key in dictForMax:
            if sum(dictForMax[key]) > maxSum:
                maxSum = sum(dictForMax[key])
                maxKey = key
        else:#After finished iterating
            plt.bar(np.arange(0,20),subjectsByBlock[maxKey], label=maxKey)
            plt.xticks(bins,tickLabels,rotation=45)
            dictForMax.pop(maxKey,None)
            
    plt.rcParams['figure.figsize'] = (25.0, 10.0)
    plt.legend(loc='upper right')
    plt.savefig(project.base_dir + '/graphs/basic-counts/most-common-subjects.png')
    plt.clf()   
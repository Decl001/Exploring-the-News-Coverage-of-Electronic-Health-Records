import os
from datetime import datetime
import TAGME_Annotation as tag_a 
import MetaDataAnalysis as mda 
import EntityNetworkGraphing as eng 
from ProjectController import Project 

if __name__ == '__main__':
    #create project if it doesn't exist in current dir
    curr_dir = os.getcwd()
    curr_dir = curr_dir.replace('\\','/')
    print(curr_dir)
    project = Project()
    if 'EHR-Project' not in os.listdir(curr_dir):
        project.create_project('EHR-Project',
                                'Analysis of impact of EHR on the U.S. Healthcare Market',
                                'Electronic health record',
                                curr_dir,
                                [curr_dir+'/EHR-20070101-20170601-nyt-articles-no-duplicates-clean-bodies.csv',
                                curr_dir+'/PHR-20070101-20170601-nyt-articles-no-duplicates-clean-bodies.csv'])
    else:
        project.load_project(curr_dir+'/EHR-Project')
    
    print('-------------------------------------------')
    print('[' + str(datetime.now()) + ']: Beginning analysis')
    mda.plot_article_counts(project)
    print('[' + str(datetime.now()) + ']: Graphed article counts')
    mda.count_subjects(project)
    print('[' + str(datetime.now()) + ']: Graphed article subjects')
    tag_a.annotate_data(project)
    print('[' + str(datetime.now()) + ']: Annotated article data')
    eng.run_all(project)
    print('[' + str(datetime.now()) + ']: Built Network Graphs')


"""
This file contains the project class,
it creates a .proj file, which specifies 
the details of a project. It has methods to
load and save projects.
"""
import os
import ast

class Project:

    def __init__(self):
        self.name = ''
        self.desc = ''
        self.datasets = []
        self.query_entity = ''
        self.base_dir = ''
    
    def create_project(self,name,desc,query_entity,base_dir,datasets):
        self.name = name
        self.desc = desc
        self.query_entity = query_entity
        self.base_dir = base_dir + '/' + name
        self.datasets = datasets
        #create the base directory
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        #create the .proj file
        with open(self.base_dir + '/' + name + '.proj', mode='w') as f:
            f.write(name + '\n')
            f.write(desc + '\n')
            f.write(query_entity + '\n')
            f.write(str(datasets) + '\n')
        #create the graph folders
        if not os.path.exists(self.base_dir + '/graphs'):
            os.mkdir(self.base_dir + '/graphs')
        if not os.path.exists(self.base_dir + '/basic-counts'):
            os.mkdir(self.base_dir + '/graphs/basic-counts')
        if not os.path.exists(self.base_dir + '/graphs/tagme-counts'):
            os.mkdir(self.base_dir + '/graphs/tagme-counts')
        if not os.path.exists(self.base_dir + '/graphs/network-graphs'):
            os.mkdir(self.base_dir + '/graphs/network-graphs')
        if not os.path.exists(self.base_dir + '/graphs/central-entities'):
            os.mkdir(self.base_dir + '/graphs/central-entities')
        #create folder for annotated data
        if not os.path.exists(self.base_dir + '/annotated-data'):
            os.mkdir(self.base_dir + '/annotated-data')

    def load_project(self,base_dir):

        fname = ''
        for name in os.listdir(base_dir):
            if '.proj' in name:
                fname = name
                break
        if fname == '':
            #throw exception if no valid file in directory
            raise FileNotFoundError('No valid projects in directory')
        else:
            self.base_dir = base_dir
            with open(base_dir + '/' + fname) as f:
                lines = [line.rstrip('\n') for line in f]
                try:
                    self.name = lines[0]
                    self.desc = lines[1]
                    self.query_entity = lines[2]
                    self.datasets = ast.literal_eval(lines[3])
                except IndexError:
                    raise RuntimeError('''Corrupt project file, try removing
                                             this project file, and creating
                                              a new project in this directory''')

    

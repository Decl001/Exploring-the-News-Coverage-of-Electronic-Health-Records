'''
This file contains the methods generated
for creating and filtering
'''
import os
import ast
import math
import nltk
import numpy as np
import networkx as nx 
import pandas as pd
import ExtractKeywordsForArticle as ex 
from nltk.tokenize import sent_tokenize
from plotly.graph_objs import *
import plotly as py
from matplotlib import pyplot as plt 
from collections import Counter
import ProjectBaseMethods as pbm 


def getConnectionKeywords(article, char_pos_start, char_pos_end):
    '''
    This method retrieves all adjectives and adverbs that are 
    contained within the sentences that connect two entites
    '''
    first_idx = 0
    second_idx = 0
    article = sent_tokenize(article)
    char_count = 0
    found_start = False
    for i,sent in enumerate(article):
        char_count += len(sent)
        if char_count >= char_pos_start and not found_start:
            first_idx = i
            found_start = True
        if char_count >= char_pos_end:
            second_idx = i
            break
    text = article[first_idx]
    if second_idx != first_idx:
        text += ' ' + article[second_idx]
    text = ex.removeStopWords(text)
    text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(text)
    return_tags = []
    for tag in tags:
        if tag[1] == 'JJ':
            return_tags.append(tag[0])
            
    return return_tags

def getArticleConnections(article, annotations):
    '''
    This method finds all connections between annotated entities
    within an article. It builds a dictionary  of the form

    {
        ENTITY_TITLE : {
            count : integer
            start : integer (character position within article)
            end : integer (character position within article)
            connections : {
                ANOTHER_ENTITY_TITLE : {
                    weight : integer
                    keywords : list (adjectives and adverbs in connecting sentence)
                }
                            .
                            .
                            .
            }
        }
    }
    '''
    node_dict = {}
    for annot in annotations:
        if annot['title'] in node_dict:
            node_dict[annot['title']]['count'] += 1
            node_dict[annot['title']]['start'].append(annot['start'])
            node_dict[annot['title']]['end'].append(annot['end'])
        else:
            this_node_data = {
                'count' : 1,
                'start' : [annot['start']],
                'end' : [annot['end']],
                'connections' : {}
            }
            node_dict[annot['title']] = this_node_data
            
    #using formula |tw| = a - b * p/|d|
    a = 120
    b = 50
    for i,key in enumerate(node_dict):
        for k,pos in enumerate(node_dict[key]['end']):#look at the ending character
            for j,otherKey in enumerate(node_dict):
                if j > i:
                    for l,otherPos in enumerate(node_dict[otherKey]['start']):
                        tw = a-b*(pos/(len(article)))
                        if (otherPos - pos) < tw and otherPos - pos > 0:
                            if otherKey in node_dict[key]['connections']:
                                node_dict[key]['connections'][otherKey]['weight'] +=1
                                node_dict[otherKey]['connections'][key]['weight'] +=1
                                node_dict[key]['connections'][otherKey]['keywords'] += getConnectionKeywords(article, pos, otherPos)
                                node_dict[otherKey]['connections'][key]['keywords'] += getConnectionKeywords(article, pos, otherPos)
                            else:
                                baseConnObject = {
                                    'weight' : 1,
                                    'keywords' : getConnectionKeywords(article, pos, otherPos)
                                } 
                                node_dict[key]['connections'][otherKey] = baseConnObject
                                node_dict[otherKey]['connections'][key] = baseConnObject
                                
    return node_dict


def mergeDictionaries(complete_dict, article_dict):
    '''
    This method merges an article dictionary with a larger
    one detailing a specific time period
    '''
    for key in article_dict:
        if key not in complete_dict:
            complete_dict[key] = article_dict[key]
        else:
            complete_dict[key]['count'] += article_dict[key]['count']
            for otherKey in article_dict[key]['connections']:
                if otherKey not in complete_dict[key]['connections']:
                    complete_dict[key]['connections'][otherKey] = article_dict[key]['connections'][otherKey]
                else:
                    complete_dict[key]['connections'][otherKey]['weight'] += article_dict[key]['connections'][otherKey]['weight']
    return complete_dict

def generate_graph(node_dict):
    '''
    This method generates a networkx graph based on a 
    dictionary that was created previously
    '''
    G = nx.Graph()
    for i,key in enumerate(node_dict):
        G.add_node(i, title=key)
        node_dict[key]['graph_index'] = i
    for key in node_dict:
        for otherKey in node_dict[key]['connections']:
            idx_1 = node_dict[key]['graph_index']
            idx_2 = node_dict[otherKey]['graph_index']
            weight = node_dict[key]['connections'][otherKey]['weight']
            G.add_edge(idx_1, idx_2, weight=weight)
    return G

def circleNewEntities(this_dict, last_dict, layout):
    '''
    This method puts a circle around entities that were
    not in the previous graph
    '''
    circles = []
    radius = 3
    for key in this_dict:
        if key not in last_dict:
            g_idx = this_dict[key]['graph_index']
            x,y = layout[g_idx]
            this_circle = {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': -1*radius + x,
                'y0': radius + y,
                'x1': radius + x,
                'y1': -1 * radius + y,
                'line': {
                    'color': '#FF4500',
                    'dash' : 'dot'
                }
            }
            circles.append(this_circle)
    return circles

def generateLayout(graph):
    return nx.spring_layout(graph)

def generateEdgeTrace(layout, graph, node_dict):
    edge_trace = []
    for edge in graph.edges:
        thisTrace = dict(type='scatter',
                       x=[layout[edge[0]][0], layout[edge[1]][0]],
                       y=[layout[edge[0]][1], layout[edge[1]][1]],
                       mode='lines',
                       line=dict(width=(0.5*node_dict[graph.nodes[edge[0]]['title']]['connections'][graph.nodes[edge[1]]['title']]['weight']),
                                 color='#999999'))
        edge_trace.append(thisTrace)
    return edge_trace

def generateNodeTrace(layout, graph, node_dict):
    node_trace = Scatter(
        x = [],
        y = [],
        text = [],
        mode='markers+text',
        textposition='bottom',
        textfont=dict(
            family='sans serif',
            size=11,
            color='#000'
        ),
        marker=Marker(
            showscale=True,
            colorscale='Bluered',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )
    for node in layout:
        node_trace['x'].append(layout[node][0])
        node_trace['y'].append(layout[node][1])
    for node in graph.nodes():
        adj_dict = node_dict[graph.nodes[node]['title']]['connections']
        n_connections = len(adj_dict)
        node_trace['marker']['color'].append(n_connections)
        node_info = graph.nodes[node]['title']
        node_trace['text'].append(node_info)

    return node_trace

def generate_path_lengths(node_dict, graph,query_entity):
    idx = node_dict[query_entity]['graph_index']
    path_lengths = nx.single_source_shortest_path_length(graph, source=idx)
    sorted_path_lengths = []
    for key in path_lengths:
        while len(sorted_path_lengths) < path_lengths[key] + 1:
            sorted_path_lengths.append([])
        sorted_path_lengths[path_lengths[key]].append(key)
    return sorted_path_lengths

def generateDynamicTextNodeTrace(pos,node_dict,graph):
    
    colours = ['#FF0000', '#AA0055', '#880088', '#5500AA', '#0000FF']
    node_trace = []
    page_rank = nx.pagerank(graph)
    minpgr = 10
    maxpgr = -1
    for key,value in page_rank.items():
        if value < minpgr:
            minpgr = value
        if value > maxpgr:
            maxpgr = value
    for i,node in enumerate(node_dict):
        #find the nearest node
        nearest_node = []
        min_distance = 1000
        for j,other_node in enumerate(node_dict):
            if j != i:
                x1 = pos[node_dict[node]['graph_index']][0]
                y1 = pos[node_dict[node]['graph_index']][1]
                x2 = pos[node_dict[other_node]['graph_index']][0]
                y2 = pos[node_dict[other_node]['graph_index']][1]
                distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = [x2,y2]
        #generate the text position based on the nearest node
        text_position = ''
        x = pos[node_dict[node]['graph_index']][0]
        y = pos[node_dict[node]['graph_index']][1] 
        
        if y > nearest_node[1]:
            text_position = 'top'
        else:
            text_position = 'botton'
        #work out what the color should be
        step = (maxpgr - minpgr)/5
        c_pos = 0
        for j in range(5):
            if (minpgr + (j+1)*step) > page_rank[node_dict[node]['graph_index']]:
                c_pos = j
                break
        this_trace = dict(type='scatter',
                         x=x,
                         y=y,
                         text=node,
                         textposition='top',
                         mode='markers+text',
                         marker=Marker(
                             size=500*page_rank[node_dict[node]['graph_index']],
                             color=colours[c_pos],
                             line=dict(width=2)
                         ),
                         textfont=dict(
                             family='sans serif',
                             size=10,
                             color='#000'
                         )
                          
                          
                    )
        node_trace.append(this_trace)
    return node_trace

def generate_ehrcentric_graph(graph, node_dict,sorted_path_lengths,count,project):
    xy_dict = {}
    circles = []
    for i,group in enumerate(sorted_path_lengths):
        if i == 0:
            xy_dict[group[0]] = {
                                    'pos' : [0,0],
                                    'distance' : 0
                                }
        else:
            #find positions on this circle
            #base radius = 5
            n_positions = len(group)
            this_radius = 5 * i
            this_pos_set = []
            this_angle = 360/n_positions
            angles = []
            for j in range(n_positions):
                angles.append(j*this_angle)
            for j,angle in enumerate(angles):
                if j == 0:
                    this_pos_set.append([0,this_radius])
                else:
                    orig_x = 0
                    orig_y = this_radius
                    x = (orig_x*math.cos(math.radians(angle))) - (orig_y*math.sin(math.radians(angle)))
                    y = (orig_y*math.cos(math.radians(angle))) + (orig_x*math.sin(math.radians(angle)))
                    this_pos_set.append([x,y])
            for j,node in enumerate(group):
                xy_dict[node] = {
                                    'pos' : this_pos_set[j],
                                    'distance'  : i
                                }
            #create this circle
            thisCircle = {
                'type': 'circle',
                'xref': 'x',
                'yref': 'y',
                'x0': -1*this_radius,
                'y0': this_radius,
                'x1': this_radius,
                'y1': -1 * this_radius,
                'line': {
                    'color': '#989898',
                    'dash' : 'dot'
                }
            }
            circles.append(thisCircle)
    node_trace = Scatter(
            x = [],
            y = [],
            text = [],
            mode='markers',
            hoverinfo='text',
            marker=Marker(
                showscale=True,
                colorscale='Bluered',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Number of connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
    for node in xy_dict:
        node_trace['x'].append(xy_dict[node]['pos'][0])
        node_trace['y'].append(xy_dict[node]['pos'][1])
    for node in xy_dict:
        adj_dict = node_dict[graph.nodes[node]['title']]['connections']
        n_connections = len(adj_dict)
        node_trace['marker']['color'].append(n_connections)
        node_info = 'Node:<br>Title: ' + graph.nodes[node]['title']
        node_info += '<br>Number of connections = ' + str(n_connections)
        node_trace['text'].append(node_info)
    fig = Figure(data=[node_trace],layout=Layout(
                    title='<br>Article Entity Network',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    width=1000,
                    height=1000,
                    shapes=circles,
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text=None,
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    py.offline.plot(fig,filename=project.base_dir+'/graphs/network-graphs/ehr-centric' 
                                                        + str(count) + '.html', auto_open=False)

def build_connections(project):
    dfs = []
    #use the annotated data
    for ds in os.listdir(project.base_dir + '/annotated-data'):
        try:
            dfs.append(pd.read_csv(project.base_dir + '/annotated-data/'+ds))
        except:
            try:
                dfs.append(pd.read_csv(project.base_dir + '/annotated-data/'+ds,encoding='cp1252'))
            except:
                raise Exception('Could not read datasets')
    
    df_node_lists = []
    for i,df in enumerate(dfs):
        annotations = []
        for val in df.TAGME_Annotations:
            if type(val) is str:
                annotations.append(ast.literal_eval(val))
            else:
                annotations.append(val)
        curr_node_dict = {}
        this_node_list = []
        for j,article in enumerate(df.article_body):
            if j != 0:
                try:
                    if pbm.isNewHalfYear(df.date[j], df.date[j-1]):
                        this_node_list.append(curr_node_dict)
                        curr_node_dict = {}
                except:
                    pass
            if annotations[j] != [] and type(article) is str:
                for k,o_df in enumerate(dfs):
                    if k<i and df.url[j] in list(o_df.url):
                        break
                else:
                    article_dict = getArticleConnections(df.article_body[j], annotations[j])
                    curr_node_dict = mergeDictionaries(curr_node_dict, article_dict)
        this_node_list.append(curr_node_dict)    
        df_node_lists.append(this_node_list)
    #generate merged node dicts
    full_node_list = [node_dict for node_dict in df_node_lists[0]]
    for node_list in df_node_lists[1:]:
        for i,node_dict in enumerate(node_list):
            full_node_list[i] = mergeDictionaries(full_node_list[i],node_dict)
    #generate network graphs
    graph_list = []
    for node_dict in full_node_list:
        G = generate_graph(node_dict)
        graph_list.append(G)
    return full_node_list,graph_list

def generate_filtered_graphs(full_node_list,graph_list,project):
    DISTANCE_THRESHOLD = 6
    CONNECTIONS_THRESHOLD = 2
    for i,node_dict in enumerate(full_node_list):
        G = graph_list[i]
        filtered_node_dict = {}
        title_list = []
        sorted_path_lengths = generate_path_lengths(node_dict,G,project.query_entity)
        for j,group in enumerate(sorted_path_lengths):
            if j > DISTANCE_THRESHOLD:
                break
            else:
                for node in group:
                    adj_dict = node_dict[G.nodes[node]['title']]['connections']
                    n_connections = len(adj_dict)
                    if n_connections >= CONNECTIONS_THRESHOLD:
                        title_list.append(G.nodes[node]['title'])
        #create filtered dict
        for key in node_dict:
            if key in title_list:
                filtered_node_dict[key] = node_dict[key]
                connections_dict = node_dict[key]['connections']
                discard_keys = []
                #discard connections to nodes not in the filtered graph
                for otherKey in connections_dict:
                    if otherKey not in title_list:
                        discard_keys.append(otherKey)
                for otherKey in discard_keys:
                    connections_dict.pop(otherKey)
                filtered_node_dict[key]['connections'] = connections_dict

        filtered_community_graph = generate_graph(filtered_node_dict)    
        pos = generateLayout(filtered_community_graph)
        edge_trace = generateEdgeTrace(pos,filtered_community_graph,filtered_node_dict)
        node_trace = generateNodeTrace(pos,filtered_community_graph,filtered_node_dict)
        fig = Figure(data=edge_trace+[node_trace],layout=Layout(
                        title='<br>Filtered Community Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text=None,
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
        py.offline.plot(fig,filename=project.base_dir+'/graphs/network-graphs/filtered-graph' 
                                                        + str(i) + '.html', auto_open=False)
def track_central_entities(full_node_list,graph_list,project):
    central_entities = []
    for i, graph in enumerate(graph_list):
        this_node_dict = full_node_list[i]
        idx = this_node_dict['Electronic health record']['graph_index']
        path_lengths = nx.single_source_shortest_path_length(graph, source=idx)
        for node in path_lengths:
            if node != idx and path_lengths[node] < 3:
                if len(this_node_dict[graph.nodes[node]['title']]['connections']) > 6:
                    central_entities.append(graph.nodes[node]['title'])

    #require central entities to satisfy the conditions twice
    count_dict = Counter(central_entities)
    final_central_entities = []
    for key,val in count_dict.items():
        if val > 1:
            final_central_entities.append(key)
    central_entities = list(final_central_entities)
    for entity in central_entities:
        page_ranks = []
        connections = []
        distance = []
        for i, graph in enumerate(graph_list):
            this_node_dict = full_node_list[i]
            if entity in this_node_dict:
                idx = this_node_dict['Electronic health record']['graph_index']
                node = this_node_dict[entity]['graph_index']
                if nx.has_path(graph,idx,node):
                    distance.append(len(nx.shortest_path(graph,idx,node)))
                else:
                    distance.append(0)
                connections.append(len(this_node_dict[entity]['connections']))
                pgr = nx.pagerank(graph)
                if node in pgr:
                    page_ranks.append(100*pgr[node])
                else:
                    page_ranks.append(0)
            else:
                page_ranks.append(0)
                connections.append(0)
                distance.append(0)
        
        #plot the graph
        N = len(page_ranks)
        ind = np.arange(N)
        width = 0.3
        fig,ax = plt.subplots()
        rects1 = ax.bar(ind, page_ranks, width, color='b')
        rects2 = ax.bar(ind+width, connections, width, color='g')
        rects3 = ax.bar(ind+width*2, distance, width, color='y')
        ax.set_title(entity)
        ax.legend((rects1[0], rects2[0], rects3[0]), ('PageRank(*100)', 'Connections', 'Distance to EHR'))
        df = []
        try:
            df = pd.read_csv(project.datasets[0])
        except:
            try:
                df = pd.read_csv(project.datasets[0],encoding='cp1252')
            except:
                raise Exception('Could not read dataset')
        tickLabels = pbm.generate_ticklabels(df)
        ax.set_xticks(ind + 2*width / 2)
        ax.set_xticklabels(tickLabels)
        plt.rcParams['figure.figsize'] = (25.0, 10.0)
        plt.legend(loc='upper right')
        plt.savefig(project.base_dir + '/graphs/central-entities/'+entity+'.png')

    

def run_all(project):
    node_list,graph_list = build_connections(project)
    for i,node_dict in enumerate(node_list):
        G = graph_list[i]
        sorted_path_lengths = generate_path_lengths(node_dict,G,project.query_entity)
        generate_ehrcentric_graph(G,node_dict,sorted_path_lengths,i,project)
    generate_filtered_graphs(node_list,graph_list,project)
    track_central_entities(node_list,graph_list,project)
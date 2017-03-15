import pickle
import numpy as np
import networkx as nx
import pandas as pd
import operator
from sklearn.preprocessing import normalize
#from brainx import util, detect_modules, modularity


#Utils for workspace files

def build_key_codes_from_workspaces(workspaces, datadir):
    """Return a list of unique Brainmap ids from a list of workspace files. 
    Function reads the individual files and builds a unique key from column one and five
    requires pandas imported as pd 
    
    
    Parameters:
    ------------
    workspaces: a python list
                list of filenames for workspaces
    datadir: string
                location of where all the workspace csvs are
    
    
    Returns:
    ----------
    key_codes: a python nested list of numeric key codes representing brainmap study+contrast unique ids
               every index is the list of key codes in the corresponding workspace. 
               length of list = number of workspaces
               length of each sublist= number of keys in workspace
    
    """
    key_codes=[]
    for x in workspaces:
        file =datadir+x[:-1]
        df = pd.read_csv(file, names = ['zero','one','two', 'three','four','five','six','seven','eight', 'nine'])
        true_only= df[df['zero']>0]
        one_col = true_only['one']
        five_col = true_only['five']
        key_codes.append(one_col.map(str).values + five_col.map(str).values)
    return key_codes

def build_keycodes_from_excel_csv(excel_csv_file):
    df=pd.read_csv(excel_csv_file)
    regionlist=df.keys()
    relabel_dict={idx:x[:] for idx, x in enumerate(regionlist)}
    
    keycodes=[]
    for key in df.keys():
        studies_in_region=df[key][pd.notnull(df[key])]
        nstudies=len(studies_in_region)
        regions_in_study_list=[]
        [regions_in_study_list.append(studies) for studies in studies_in_region]
        keycodes.append(regions_in_study_list)
    return keycodes


def build_jaccard(key_codes):
    array_dims=len(key_codes), len(key_codes)
    jaccard=np.zeros(shape=array_dims)
    for col_idx, x in enumerate(key_codes):
        for row_idx, y in enumerate(key_codes):
            intersect=float(len(set(x) & set(y)))
            union=float(len(list(set(x) | set(y))))
            if union:
                jaccard[col_idx, row_idx]=intersect/union
    return jaccard
    

def build_n_coactives_array(key_codes):
    array_dims=len(key_codes), len(key_codes)
    n_coactives_array =np.zeros(shape=array_dims)
    
    for col_idx, x in enumerate(key_codes):
        for row_idx, y in enumerate(key_codes):
            n_coactives_array[col_idx, row_idx]=len(set(x) & set(y))
    return n_coactives_array
    
def normalize_n_coactives_array(n_coactives_array):
    #Use sklearn's normalize function w L2 remove diagonal first
    norm_coactives_array=n_coactives_array.copy()
    np.fill_diagonal(norm_coactives_array, 0)
    
    return norm_coactives_array

def build_region_labels_dict(regionlist):
    relabel_dict ={idx:x[:-5] for idx, x in enumerate(regionlist)}
    return relabel_dict
 
 
 
 
# Working with networkx graphs
    
def applycost_to_g(G,cost):
    """Threshold graph to achieve cost.

    Return the graph after the given cost has been applied. Leaves weights intact.

    Parameters
    ----------
    G: input networkx Graph

    cost: float
        Fraction or proportion of all possible undirected edges desired in the
        output graph.


    Returns
    -------
    G_cost: networkx Graph
        Networkx Graph with cost applied

    threshold: float
        Correlations below this value have been set to 0 in
        thresholded_corr_mat."""
        
    Complete_G=nx.complete_graph(G.number_of_nodes())
    n_edges_keep=int(Complete_G.number_of_edges()*cost)
    weights=[(G[x][y]['weight']) for x,y in G.edges_iter()]
    sorted_weights=sorted(weights, reverse=1)
    thresh=sorted_weights[n_edges_keep+1]
    remove_edgelist=[(x,y) for x,y in G.edges_iter() if G[x][y]['weight']<thresh]
    G_cost=G.copy()
    G_cost.remove_edges_from(remove_edgelist)
    return G_cost

def remove_edges_by_weight(G, max_weight):
    G_removed=G.copy()
    for x,y in G.edges_iter():
        if G_removed[x][y]['weight']<max_weight:
            G_removed.remove_edge(x,y)
            
    for x,y in G.degree_iter():
        if y<2:
            G_removed.remove_node(x)
    return G_removed 

def remove_weight_edge_attribute(G):
    G_binary=G.copy()
    for x,y in G_binary.edges_iter():
        del G_binary[x][y]['weight']
    return G_binary


def _remove_edgeless_nodes(G):
    to_remove=[]
    degree = G.degree()
    for x in degree:
        if degree[x]==0:
            to_remove.append(x)

    G.remove_nodes_from(to_remove)
    return G

def build_binarized_graph(G):
    """Takes graph converts it to a np array, binarizes it, builds networkx graph with binary edges"""
    
    binary_array=nx.to_numpy_matrix(G)
    binary_array=np.where(binary_array>0, 1,0)
    binary_G=nx.from_numpy_matrix(binary_array)
    #if nx.is_connected(binary_G)==0:
     #   print "Graph is not connected, removing nodes"
      #  binary_G=_remove_edgeless_nodes(binary_G)
    #else:
#        print "Graph is connected"
    return binary_G
    

def domain_filter_keycodes(key_codes, studies_filtered_by_domain, domain):
    """ filters a keycodes list by behavioral domain
        
        Parameters
        ------------
    
        key_codes: list of lists
            list of key_codes lists by region
    
        studies_filtered_by_domain: dict
            keys = behavioral domain, values are key_codes that correspond to that behavioral domain 
        
        domain: selected behavioral domain string
        Includes: 'Memory', 'Working Memory', 'Emotion', 'Attention', 'Language', 'Vision', 'Audition'

    Returns
    ------------
    domain_filtered_codes: list of lists
                    list of key_codes lists by region filtered by a particular domain
    
    """
    domainlist=studies_filtered_by_domain[domain]
    domain_filtered_codes=[set(x) & set(domainlist) for x in key_codes]
    return domain_filtered_codes
    

def run_basic_metrics(G, top_n=5):
    """runs a bunch of basic metrics and returns a dict"""
    basic_metrics=dict()
    basic_metrics['degrees']=nx.degree(G)
    basic_metrics['cpl']=[nx.average_shortest_path_length(g) for g in nx.connected_component_subgraphs(G) if g.number_of_nodes()>1]
    basic_metrics['ccoeff']=nx.clustering(G)
    basic_metrics['degree_cent']=nx.degree_centrality(G)
    basic_metrics['between_cent']=nx.betweenness_centrality(G)
    for x in ['degrees','degree_cent','between_cent','ccoeff']:
        sorted_x = sorted(basic_metrics[x].items(), key=operator.itemgetter(1))
        tops=[]
        for y in range(top_n):
            tops.append(sorted_x[-top_n:][y][0])
        basic_metrics['top'+x]=tops
    
    return basic_metrics

def run_weighted_metrics(G, top_n=5):
    """runs a bunch of basic metrics and returns a dict"""
    weighted_metrics=dict()
    weighted_metrics['degrees']=nx.degree(G, weight='weight')
    weighted_metrics['cpl']=[nx.average_shortest_path_length(g, weight='weight') for g in nx.connected_component_subgraphs(G) if g.number_of_nodes()>1]
    weighted_metrics['ccoeff']=nx.clustering(G, weight='weight')
    weighted_metrics['degree_cent']=nx.degree_centrality(G)
    weighted_metrics['between_cent']=nx.betweenness_centrality(G)
    for x in ['degrees','degree_cent','between_cent','ccoeff']:
        sorted_x = sorted(weighted_metrics[x].items(), key=operator.itemgetter(1))
        tops=[]
        for y in range(top_n):
            tops.append(sorted_x[-top_n:][y][0])
        weighted_metrics['top'+x]=tops
    
    return weighted_metrics
    
    
def build_influence_matrix(n_coactives_array):
    n_coactive_mat=n_coactives_array.copy()
    diagonal=n_coactive_mat.diagonal()
    a=n_coactive_mat/diagonal[:, np.newaxis] #dividing by row
    b=n_coactive_mat/diagonal #dividing by column
    influence_mat=a-b # positive: rows influence column (B infl. A) , negative: col influence row (A inf. B)
    #influence_mat=np.triu(influence_mat)
    return influence_mat
    
def build_influence_digraph(n_coactives_array):
    influence_mat=build_influence_matrix(n_coactives_array)
    influence_di_mat=influence_mat*(influence_mat>0)
    influence_diG=nx.DiGraph(influence_di_mat)
    return influence_diG
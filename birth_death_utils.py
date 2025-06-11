import os
import functools
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import pickle
from tqdm import tqdm

def leaves(graph):
    """
    Returns the terminal leaves of a tree

    Parameters
    ----------
    graph : nx.DiGraph()
        tree (directed acyclic graph)

    Returns
    -------
    list : list of node objects
        list of node labels of leaves

    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]

def internal_nodes(graph):
    """
    Returns the non-leaves node of a tree

    Parameters
    ----------
    graph : nx.DiGraph()
        tree (directed acyclic graph)

    Returns
    -------
    list : list of node objects
        list of internal nodes of a graph

    """
    return [n for n in graph.nodes() if not n in leaves(graph)]

def root(graph):
    """
    Returns the root of a tree
    
    Parameters
    ----------
    graph : nx.DiGrap()
        tree (directed acyclic graph)

    Returns
    -------
    n : node object
        label of the root of graph

    """
    for n in graph.nodes():
        if graph.in_degree(n) == 0:
            return n

def witness_nb(graph):
    return list(nx.get_node_attributes(graph, 'state').values()).count(True)


def subtree_size(graph, node):
    """
    Returns the size of the subtree of graph with given root node

    Parameters
    ----------
    graph : nx.DiGraph()
        tree
    node : root of the subtree

    Returns
    -------
    int
        Size of the subtree

    """
    return len(dfs_tree(graph,node).nodes())


def generate_tree(lda, mu, Nact, Ninact):
    """
    Generate a tree (arbre réel) according to birth death model.

    Parameters
    ----------
    lda : float
        birth rate of new node per node per iteration
    mu : float
        death rate of nodes per node per per iteration
    Nact : int
        number of iterations of the active reproduction phase
    Ninact : int
        number of iterations of the pure death phase (lda is set to 0)

    Returns
    -------
    G : nx.DiGraph()
        networkx graph object of the generated tree with following node attributes:
            'state' : boolean, True if node living at the end of simulation
            'birth_time' : int
            'death_time' : int

    """
    currentID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living = {0:True}

    birth_time = {0:0}
    death_time = {}

    pop = 1
    prob_birth = lda
    prob_death = mu

    for t in range(Nact):
        for current_node in list(G.nodes()):
            r = np.random.rand()
            if r < prob_birth and living[current_node]:
                currentID += 1
                G.add_node(currentID)
                G.add_edge(current_node, currentID)
                living[currentID] = True
                pop += 1
                birth_time[currentID] = t
            if prob_birth < r and r < (prob_birth + prob_death) and living[current_node]:
                living[current_node] =  False
                pop -= 1
                death_time[current_node] = t
        if pop == 0:
            break

    for t in range(Ninact):
        for current_node in list(G.nodes()):
            r = np.random.rand()
            if r <  prob_death and living[current_node]:
                living[current_node] =  False
                pop -= 1
                death_time[current_node] = t + Nact
            if pop == 0:
                break

    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')
    return G

def generate_stemma(GG):
    """
    Returns the stemma of a tradition generated from generate_tree

    Parameters
    ----------
    GG : nx.DiGraph()
        tree object with at least node attributes 'state' given

    Returns
    -------
    G : nx.DiGraph()
        stemma obtained from GG.

    """
    G = nx.DiGraph(GG)
    living = {n: G.nodes[n]['state'] for n in list(G.nodes())}
    
    # recursivelly remove dead leaves until all terminal nodes are living witnesses
    
    terminal_dead_nodes = [n for n in leaves(G) if not living[n]]
    while terminal_dead_nodes != []:
        for n in terminal_dead_nodes:
            G.remove_node(n)
        terminal_dead_nodes = [n for n in leaves(G) if not living[n]]

    # remove non-branching consecutive dead nodes
    
    unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.out_degree(n) == 1 and G.in_degree(n) == 1]
    while unwanted_virtual_nodes != []:
        for n in unwanted_virtual_nodes:
            G.add_edge(list(G.predecessors(n))[0], list(G.neighbors(n))[0])
            G.remove_node(n)
        unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.in_degree(n) == 1 and G.out_degree(n) ==1]

    if not living[root(G)] and G.out_degree(root(G)) == 1:
        G.remove_node(root(G))
        
    return G

def draw_tree(G, filename):
    """
    Draw a tree generated from generate_tree or generate_stemma as svg file 
    with living nodes colored in red and grey dead nodes

    Parameters
    ----------
    G : nx.DiGraph()
        tree object with at least 'state' attribute given
    filename : string
        

    Returns
    -------
    None.

    """
    living = nx.get_node_attributes(G, 'state')
    color_map = {node: 'red' if state else 'grey' for node, state in zip(living.keys(), living.values())}
    nx.set_node_attributes(G, color_map, name='color')
    nx.nx_pydot.write_dot(G, 'graph.dot')
    os.system('dot -Tsvg graph.dot > {}.svg'.format(filename))
    
def csv_dump(G, filename):
    """
    Generate a csv file representing a tree obtained from generate_tree with layout
                
                label, parent, birth_time, death_time
                
                parent is -1 if node is root
                death_time is -1 if node living at the end of simulation
    """
    out = 'label,parent,birth_time,death_time\n'
    for n in G.nodes():
        bt = G.nodes[n]['birth_time']
        dt = G.nodes[n]['death_time'] if not G.nodes['state'] else -1
        pred = G.predecessors(n)
        par = pred[0] if pred != [] else -1
        out.append(f'{n},{par},{bt},{dt}\n')
    with open(f'{filename}.csv', 'w') as f:
        f.write(out)
        
def plot_heatmap(data, title, xticks, yticks, precision, cmap='viridis'):
    '''
    Plots 2D phase diagrams with printed values

    Parameters
    ----------
    data : 2D array
        data to be plotted
    title : string
        title of the plot
    xticks : 1D float array
        list of ticks values displayed on x-axis
    yticks : 1D float array
        list of ticks values displayed on y-axis
    precision : int
        Nb of significant digits printed on plots
        
    Returns
    -------
    None.
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap=cmap)
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            text = ax.text(j, i, f'%.{precision}f'%data[i][j],
                           ha="center", va="center", color="w")
            
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    
    ax.set_xlabel(r'$\lambda ~~~~ (10^{-3})$')
    ax.set_ylabel(r'$\mu ~~~~ (10^{-3})$')
     
    plt.colorbar(im)
    ax.set_title(title)
    plt.show()
    
def plot_phase_diagram(path, lambda_labels, mu_labels, trad_nb, f, title, prec=2):
    data = []
    for i in mu_labels:
        line = []
        for j in lambda_labels:
            trees = []
            for k in range(trad_nb):
                g = pickle.load(open(f'{path}/lambda={j}_mu={i}/{k}', 'rb'))
                trees.append(g)
            line.append(f(trees))
        data.append(line)
    plot_heatmap(data, title, lambda_labels, mu_labels, prec)

def bootstraped(estimator):
    '''
    Decorator function returning the bootstraped mean and 10% confidence interval of an estimator

    Parameters
    ----------
    estimator : function(1D-array) -> float

    Returns
    -------
    (float,float,float)
        (bootstrap mean, 5th centile of bootstrap distribution , 95th centile)

    '''
    @functools.wraps(estimator)
    def wrapper_bootstraped(dataset):
        sample_size  = len(dataset)
        sample_nb = 100
        
        list_estimator = []
        
        progress = tqdm(total = sample_nb)
        for k in range(sample_nb):
            sample = np.random.choice(np.asarray(dataset, dtype ="object"), sample_size, replace = True)
            list_estimator.append(estimator(sample))
            progress.update(1)
            
        estimated_beta = np.mean(list_estimator)
        lb = np.quantile(list_estimator, np.linspace(0,1,20))[0]
        ub = np.quantile(list_estimator, np.linspace(0,1,20))[19]
        return (estimated_beta, lb, ub)
    return wrapper_bootstraped


def count_direct_filiation(g):
    """
    Count the number of edges that are direct filiations between surviving witnesses in an nx.DiGraph() object with 'state' node attribute given
    """
    c = 0
    for n in g.nodes():
        if not n == root(g):
            father = list(g.predecessors(n))[0]
            if g.nodes[n]['state'] and g.nodes[father]['state']:
                c += 1
    return c

def findsubsets(s, n):
    return list(itertools.combinations(set(s), n))

def rooted_triplet(graph, l):
    '''
    Returns the subtree of a given graph generated by a triplet of leaves

    Parameters
    ----------
    graph : nx.DiGraph() object
    l : list of nodes

    Returns
    -------
    st : nx.DiGraph() object

    '''
    G = nx.DiGraph(graph)
    in_subtree = {}
    for n in G.nodes():
        if n in l:
            in_subtree[n] = True
        else:
            in_subtree[n] = False
    nx.set_node_attributes(G, in_subtree, 'state')
    st = generate_stemma(G)
    return st


def imbalance_proportion(graph):
    '''
    Returns the proportion of imbalanced subtree among all 3-leaves subtrees of a graph

    Parameters
    ----------
    graph : nx.DiGraph() object

    Returns
    -------
    float : number of imbalance subtrees / total number of triplets of leaves in graph

    '''
    Q3 = {0 : nx.from_edgelist([(0,1), (0,2), (0,3)], create_using=nx.DiGraph),
          1 : nx.from_edgelist([(0,1), (1,2), (1,3), (0,4)], create_using=nx.DiGraph)}
    triplets = findsubsets(leaves(graph), 3)
    h = np.full(2, 0)
    for q in triplets:
        for i in range(2):
            if nx.is_isomorphic(rooted_triplet(graph, q), Q3[i]):
                h[i] += 1
    return h[0]/len(triplets)

def generate_yule_tree(lda, gamma, mu, Nact, Ninact):
    """
    Returns a networkx graph of the tree of a single Yule tradition starting from a single living node.

    Parameters
    ----------
    lda : float
        birth rate of nodes (simple copy)
    gamma :float
        birth rate of nodes with different species than parent (speciation rate)
    mu : float
        death rate of nodes
    Nact : int
        duration of the active reproduction phase
    Ninact : int
        duration of the decimation^ (pure death phase)

    Returns
    -------
    G : networkx.DiGraph
        directed graph representing the tradition generated from the model, with node attributes:
            state : boolean
            birth_time : int
            death_time : int
            specie : int

    """
    currentID = 0
    currentSpecID = 0
    G = nx.DiGraph()
    G.add_node(currentID)
    living = {0:True}
    spec = {0:0}
    
    birth_time = {0:0}
    death_time = {}
    
    pop = 1
    
    for t in range(Nact):
        for current_node in list(G.nodes()):
            r = np.random.rand()
            if r < lda and living[current_node]:
                currentID += 1
                G.add_node(currentID)
                G.add_edge(current_node, currentID)
                living[currentID] = True
                spec[currentID] = spec[current_node]
                pop += 1
                birth_time[currentID] = t
            if lda < r and r < lda + gamma:
                currentSpecID +=1
                currentID +=1
                G.add_node(currentID)
                G.add_edge(current_node, currentID)
                living[currentID] = True
                spec[currentID] = currentSpecID
                pop +=1                
                birth_time[currentID] = t
            if lda + gamma < r and r < (lda + mu + gamma) and living[current_node]:
                living[current_node] =  False
                pop -= 1
                death_time[current_node] = t
        if pop == 0:
            break

    survival_prob = np.exp(-mu*Ninact)
    for n,s in living.items():
        if s:
            r = np.random.rand()
            if r >= survival_prob:
                living[n] =  False
                death_time[current_node] = Nact + 1
            
    nx.set_node_attributes(G, living, 'state')
    nx.set_node_attributes(G, birth_time, 'birth_time')
    nx.set_node_attributes(G, death_time, 'death_time')
    nx.set_node_attributes(G, spec, 'specie')
    return G


def generate_stemma_yule(GG):
    """
    Generate the stemma corresponding to an 'arbre réel' generated from generate_yule_tree

    Parameters
    ----------
    GG : networkx.DiGraph
        nx directed graph instance with following node attributes specified:
            state : bool
            specie : int

    Returns
    -------
    G : networkx.DiGrap
        nx directed graph instance representing corresponding stemma with nodes labeled and colored according to specie

    """
    G = nx.DiGraph(GG)
    living = {n: G.nodes[n]['state'] for n in list(G.nodes())}
    original_label = {n:str(n) for n in GG.nodes()}
    species = nx.get_node_attributes(GG, 'specie')
    
    # recursivelly remove dead leaves
    terminal_dead_nodes = [n for n in leaves(G) if not living[n]]
    while terminal_dead_nodes != []:
        for n in terminal_dead_nodes:
            G.remove_node(n)
        terminal_dead_nodes = [n for n in leaves(G) if not living[n]]

    # remove adjacent non-branching dead nodes
    unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.out_degree(n) == 1 and G.in_degree(n) == 1]
    while unwanted_virtual_nodes != []:
        for n in unwanted_virtual_nodes:
            G.add_edge(list(G.predecessors(n))[0], list(G.neighbors(n))[0])
            G.remove_node(n)
        unwanted_virtual_nodes = [n for n in list(G.nodes()) if living[n] == False and G.in_degree(n) == 1 and G.out_degree(n) ==1]

    if not living[root(G)] and G.out_degree(root(G)) == 1:
        G.remove_node(root(G))
    
    # set color and label according to species
    color_palette = list(np.loadtxt('svg_colors.csv', dtype = str))
    living_species = set([s for n,s in species.items() if living[n]])
    color_choice = {s:np.random.choice(color_palette) for s in living_species}
    color_map = {node: color_choice[species[node]] if species[node] in living_species else 'grey' for node in G.nodes()}
    label_map = {node: '' if not state else original_label[node] for node, state in zip(living.keys(), living.values())}
    nx.set_node_attributes(G, color_map, name='color')
    nx.set_node_attributes(G, label_map, name='label')
    return G

def draw_tree_yule(G, filename):
    """
    Draw a tree with pydot as svg file in CWD

    Parameters
    ----------
    G : networkx.DiGraph
        color node attribute must be specified
    filename : string
                name or relative path of svg file created in current working directory

    Returns
    -------
    None.
    """
    color_map = nx.get_node_attributes(G, 'color')    
    nx.set_node_attributes(G, {n:'filled' for n in G.nodes()}, 'style')
    nx.set_node_attributes(G, {n: color_map[n] for n in G.nodes()}, 'fillcolor')
    g = nx.nx_pydot.to_pydot(G)
    g.write_svg(f'{filename}.svg')

def generate_yule_pop(LDA, lda, gamma, mu, Nact = 1000, Ninact = 1000, n_init = 0):
    """
    Returns a list of abundance data for the surviving witnesses of a Yule model simulation
    
    Parameters
    ----------
    LDA : float 
        birth rate of independant trees/metatradition
    lda : float
        birth rate of nodes (simple copy)
    gamma :float
        birth rate of nodes with different species than parent (speciation rate)
    mu : float
        death rate of nodes
    Nact : int
        duration of the active reproduction phase
    Ninact : int
        duration of the decimation (pure death phase)
    n_init : int
        number of initialy living independant nodes at t=0
    
    Returns
    -------
    witness_nb : list of ints 
        list of the final number of surviving witnesses for each extant works
    """
    currentID = n_init
    currentSpecID = n_init
    living = list(range(n_init))
    spec = {n:n for n in range(n_init)}
    for t in range(Nact):
        if np.random.rand() < LDA:
            currentSpecID +=1
            currentID +=1
            living.append(currentID)
            spec[currentID] = currentSpecID
            
        births = []
        deaths = []
        for current_node in living:
            r = np.random.rand()
            if r < lda:
                currentID += 1
                births.append(currentID)
                spec[currentID] = spec[current_node]
            if lda < r and r < lda + gamma:
                currentSpecID +=1
                currentID +=1
                births.append(currentID)
                spec[currentID] = currentSpecID
            if lda + gamma < r and r < (lda + mu + gamma):
                deaths.append(current_node)
        
        for n in births:
            living.append(n)
        for n in deaths:
            living.remove(n)

    deaths = []
    for node in living:
        r = np.random.rand()
        if r > np.exp(- mu * Ninact):
            deaths.append(node)
    
    for node in deaths:
        living.remove(node)
            
    witnesses = [spec[n] for n in living]
    witness_nb = list(Counter(witnesses).values())
    return witness_nb
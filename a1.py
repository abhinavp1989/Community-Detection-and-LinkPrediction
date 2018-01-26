# coding: utf-8

# # CS579: 
#
# We'll implement community detection and link prediction algorithms using Facebook "like" data.
#
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
#
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#



from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
    """
    Create the example graph from class. Used for testing.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from 
                       the root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###TODO
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    min=10000
    count_paths=0
    q = deque()
    q.append(root)
    seen = set()
    parents =[]
    
    #print (graph.neighbors('F'))
    while len(q) > 0:
        n = q.popleft()
        if n not in seen:
            
            if n==root:
                node2distances[n]=0
                node2num_paths[n]=1
                seen.add(n)
                
            else:
                for edges in graph.neighbors(n):
                    if edges in seen and min>=node2distances[edges]:
                        min=node2distances[edges]
                        
                for edges in graph.neighbors(n):
                    if edges in seen and min==node2distances[edges] :
                        count_paths+=1
                        parents.append(edges)
                        #print (parents)
                        count = min+1
                        #print (count)
                        seen.add(n)
                        
                        node2distances[n] =count
                        node2num_paths[n] =count_paths
                    
                    if n not in node2parents:
                        node2parents[n] = parents
                min=1000
                count_paths=0
                count=0
        #print(node2distances)
        parents=[]
        
        if node2distances[n]<max_depth:        
            for nn in graph.neighbors(n):
                if nn not in seen:
                    q.append(nn)
                
            
    return node2distances,node2num_paths,node2parents
    pass


def complexity_of_bfs(V, E, K):
    """
    If V is the number of vertices in a graph, E is the number of
    edges, and K is the max_depth of our approximate breadth-first
    search algorithm, then what is the *worst-case* run-time of
    this algorithm? As usual in complexity analysis, you can ignore
    any constant factors. E.g., if you think the answer is 2V * E + 3log(K),
    you would return V * E + math.log(K)
    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    """
    ###TODO
    return(V+E+math.log(K))
    pass


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    ###TODO
    max=0
    maxdepth=0
    
    credit_nodes={}
    edge_value ={}
    
    e_tup=()
    node_list =[]
    for key,value in node2distances.items():
        if max<value:
            max=value
            maxdepth=value
            
    
    
    
    
    while max>=0:
        if max==maxdepth:
            for key,value in node2distances.items():
                if value==max:
                    node_list.append(key)
                    credit_nodes[key]=1
            
        else:
            for key,value in node2distances.items():
                if value==max:
                    node_list.append(key)
                    sum=0
                    for edges,weight in edge_value.items():
                        if edges[0]==key or edges[1]==key:
                            sum+=weight
                    credit_nodes[key]=1+sum
            
            
        
        for key,values in credit_nodes.items():
            if key in node2parents:
                edge_weight= credit_nodes[key]/node2num_paths[key]
                for parents in node2parents[key] :
                
                    if key>parents:
                        e_tup = (parents,key)
                        edge_value[e_tup]=edge_weight
                    else:
                        e_tup =(key,parents)
                        edge_value[e_tup]=edge_weight
        #print (credit_nodes)            
        #print (edge_value)
        
            
            
        max-=1
        credit_nodes={}
        node_list =[]
        
    return edge_value
    pass


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    Call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """
    ###TODO
    i=0
    edge_val={}
    edge_val1={}
    for nodes in graph.nodes():
        root=nodes
        #print(root)
        #print(max_depth)
        node1distances,node1num_paths,node1parents=bfs(graph,root,max_depth)
        #print (node2distances)
        #print (node2num_paths)
        #print (node2parents)
        edge_val=bottom_up(root, node1distances, node1num_paths, node1parents)
        if i==0:
            edge_val1=edge_val
            i+=1
        else:
            for edges,values in edge_val.items():
                if edges in edge_val1:
                    edge_val1[edges]+=values
                else:
                    edge_val1[edges]=values
                    
    
    for edges,values in edge_val1.items():
        edge_val1[edges]=values/2
    #print (edge_val1)
    return edge_val1
    pass


def is_approximation_always_right():
    """
    Look at the doctests for approximate betweenness. In this example, the
    edge with the highest betweenness was ('B', 'D') for both cases (when
    max_depth=5 and max_depth=2).

    Consider an arbitrary graph G. For all max_depth > 1, will it always be
    the case that the edge with the highest betweenness will be the same
    using either approximate_betweenness verses the exact computation?
    Answer this question below.

    In this function, return either the string 'yes' or 'no'.
    >>> s = is_approximation_always_right()
    >>> type(s)
    <class 'str'>
    """
    ###TODO
    return('No')
    pass


def partition_girvan_newman(graph, max_depth):
    """
    Use approximate_betweenness implementation to partition a graph.
    Remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple components are created.

    Compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    Used Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """
    ###TODO
    edge_list=[]
    copy_g = graph.copy()
    edge_value = approximate_betweenness(copy_g, max_depth)
    #print(edge_value)
    max_val=max(edge_value.values())
    #print(max_val)
    
    
    components = [c for c in nx.connected_component_subgraphs(copy_g)]
    #print(len(components[0].nodes()))
    while(len(components)==1):
        max_val=max(edge_value.values())
        #print(max_val)
        for edges,values in edge_value.items():
            if (values==max_val):
                edge_list.append(edges)
        edge_list = (sorted(edge_list))
        #print(edge_list)
        copy_g.remove_edge(*edge_list[0])
        #print ((edge_list[0]))
        components = [c for c in nx.connected_component_subgraphs(copy_g)]
        del edge_value[edge_list[0]]
        #print('A')
        #print(len(components))
        edge_list=[]
    
    
    components = sorted(components, key=lambda x: sorted(x.nodes())[0])        
    return components
    #print (len(components))
    #print ('components sizes=', [len(c) for c in components])
    
    
    pass

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    Used this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    ###TODO
    S = nx.Graph()
    degree_of_vertex = 0
    for vertex in graph.nodes():
        degree_of_vertex = graph.degree(vertex)
        if min_degree<= degree_of_vertex:
            S.add_node(vertex)
    #print (graph.edges())
    
    for edges in graph.edges():
        if (edges[0]) in S and edges[1] in S:
            S.add_edge(edges[0],edges[1])
            
    return S    
    pass


""""
Compute the normalized cut for each discovered cluster.
"""

def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph

    >>> volume(['A', 'B', 'C'], example_graph())
    4
    """
    ###TODO
    copy_g = graph.copy()
    copy_g.remove_nodes_from(nodes)
    #print(len(graph.edges())-len(copy_g.edges()))
    return len(graph.edges())-len(copy_g.edges())
    
    pass


def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.

    >>> cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    1
    """
    ###TODO
    cut_set_count=0
    for edges in graph.edges():
       
        if edges[0] in S and edges[1] in T or edges[0] in T and edges[1] in S:
            cut_set_count+=1
    #print(cut_set_count)
    return (cut_set_count)
    pass


def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T. (See lec06.)
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value

    """
    ###TODO
    return ((cut(S,T,graph)/volume(S,graph))+(cut(S,T,graph)/volume(T,graph)))
    pass


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method,
    run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Note:Smaller norm_cut scores correspond to better partitions.

    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman

    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """
    ###TODO
    score_max_d =[]
    maxdepth_score_tuple=()
    copy_graph=graph.copy()
    
    
    for i in max_depths:
        
        comp=partition_girvan_newman(copy_graph, i)
        S=comp[0].nodes()
        T=comp[1].nodes()
        #print(len(comp[0].nodes()))
        #print(len(comp[1].nodes()))
        #print (len(comp))
        norm_cut_val = norm_cut(S,T,graph)
        maxdepth_score_tuple=(i,norm_cut_val)
        score_max_d.append(maxdepth_score_tuple)
        
    #print (score_max_d)    
    return score_max_d
    pass


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    We'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.

    *Copy* the input graph prior to removing edges.

    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.

    In this doctest, we remove edges for two friends of D:
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
    """
    ###TODO
    copy_graph=graph.copy()
    neighbors = sorted(copy_graph.neighbors(test_node))
    #print (len(neighbors))
    #print(neighbors[:n])
    i=0
    while(i<n):
        copy_graph.remove_edge(test_node,neighbors[i])
        i+=1
    #print (len(copy_graph.neighbors(test_node)))
    
    return copy_graph    
    pass



def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note: Don't return scores for edges that already appear in the graph.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
    """
    ###TODO
    #jaccard_score =[]
    neighbors = set(graph.neighbors(node))
    #print(graph.neighbors(node))
    scores = []
    #print(len(graph.nodes()))
    for n in graph.nodes():
        if n!=node and not graph.has_edge(node,n):
            neighbors2 = set(graph.neighbors(n))
            scores.append(((node,n), len(neighbors & neighbors2) /len(neighbors | neighbors2)))
    scores =sorted(scores, key=lambda x: (-x[1],x[0][1]))
    scores = scores[:k] 
    return scores
    pass


# One limitation of Jaccard is that it only has non-zero values for nodes two hops away.
#
# Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:
#
# $$
# s(x,y) = \beta^i n_{x,y,i}
# $$
#
# where
# - $\beta \in [0,1]$ is a user-provided parameter
# - $i$ is the length of the shortest path from $x$ to $y$
# - $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$


def path_score(graph, root, k, beta):
    """
    Computed a new link prediction scoring function based on the shortest
    paths between two nodes, as defined above.

    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.

    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edge (D, F) from the
    example graph. The top two edges to add according to path_score are
    (D, F), with score 0.5, and (D, A), with score .25. (Note that (D, C)
    is tied with a score of .25, but (D, A) is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = g.copy()
    >>> train_graph.remove_edge(*('D', 'F'))
    >>> path_score(train_graph, 'D', k=4, beta=.5)
    [(('D', 'F'), 0.5), (('D', 'A'), 0.25), (('D', 'C'), 0.25)]
    """
    ###TODO
    
    
    node3distances = defaultdict(int)
    node3num_paths = defaultdict(int)
    
    min=10000
    count_paths=0
    q = deque()
    q.append(root)
    seen = set()
    parents =[]
    
    #print (graph.neighbors('F'))
    while len(q) > 0:
        n = q.popleft()
        if n not in seen:
            
            if n==root:
                node3distances[n]=0
                node3num_paths[n]=1
                seen.add(n)
                
            else:
                for edges in graph.neighbors(n):
                    if edges in seen and min>=node3distances[edges]:
                        min=node3distances[edges]
                        
                for edges in graph.neighbors(n):
                    if edges in seen and min==node3distances[edges] :
                        count_paths+=1
                        parents.append(edges)
                        #print (parents)
                        count = min+1
                        #print (count)
                        seen.add(n)
                        
                        node3distances[n] =count
                        node3num_paths[n] =count_paths
                    
                    #if n not in node2parents:
                     #   node2parents[n] = parents
                min=1000
                count_paths=0
                count=0
        
        #parents=[]
        
        #if node3distances[n]<max_depth:        
        for nn in graph.neighbors(n):
            if nn not in seen:
                q.append(nn)
                
    #print(node3distances) 
    list_tup=[]
    for key,values in node3distances.items():
        if key!=root and not graph.has_edge(root,key):
            if key in node3num_paths:
                tot = (beta**node3distances[key])*node3num_paths[key]
                #print (tot)
                list_tup.append(((root,key),tot))
        
    #print(list_tup)
    scores =sorted(list_tup, key=lambda x: (-x[1],x[0][1]))
    scores = scores[:k] 
    #print (scores)
    return scores
    pass


def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.

    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph

    Returns:
      The fraction of edges in predicted_edges that exist in the graph.

    In this doctest, the edge ('D', 'E') appears in the example_graph,
    but ('D', 'A') does not, so 1/2 = 0.5

    >>> evaluate([('D', 'E'), ('D', 'A')], example_graph())
    0.5
    """
    ###TODO
    edges_in_graph_count=0
    edges_total=0
    for edges in predicted_edges:
        if edges[1] in graph.neighbors(edges[0]):
            edges_in_graph_count+=1
            edges_total+=1
        else:
            edges_total+=1
    return(edges_in_graph_count/edges_total)
    pass


"""
Download a real dataset to see how the algorithm performs.
"""
def download_data():
    """
    Download the data. Done for you.
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    """
    This takes ~10-15 seconds to run.
    """
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    #approximate_betweenness(example_graph(), 1)
    #norm_cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    #partition_girvan_newman(example_graph(), 2)
    #score_max_depths(subgraph, range(1,5))
    
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())
    
    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))

    
    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
         evaluate([x[0] for x in jaccard_scores], subgraph))
    
    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
         evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()

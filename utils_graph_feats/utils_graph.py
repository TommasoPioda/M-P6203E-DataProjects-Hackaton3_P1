import pandas as pd
import networkx as nx
import random


def network_creation(df: pd.DataFrame) -> nx.DiGraph:
    """
    Create from from scratch the respective network graph given the dataframe
    
    This function take the paper IDs and create connections based on the
    references between the papers
    
    Args:
        df (pd.Dataframe): dataframe containing:
            - "id": the identifier of the respective paper
            - "references": list of papers IDs
         
    Returns:
        nx.DiGraph: A directed graph object where nodes are paper IDs and 
        edges represent the citations.
    """
    edges = []

    # iterate over all the papers and save the connections
    for _, row in df.iterrows():
        paper = row["id"]
        for ref in row["references"]:
            edges.append((paper, ref))

    # create the graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    
    return graph

def target_definition(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Generate a new Dataframe containing pair of papers
    with respective target labels
    
    This funtion save the edges that are present in the network and create
    new unique pairs that have no reference at all between each others.
    The method concatenates the two dataframe and assign the corresponding labels:
    - 1 (connection)
    - 0 (no connection)
    
    Args:
        graph (nx.DiGraph): Direct graph that contain the papers as nodes
            and citations as edges
            
    Returns:
        concat_df (pd.DataFrame): Panda DataFrame containing:
            - "node_a": Paper that cite
            - "node_b": Cited paper
            - "Target": define if paper A cite paper B
                - 1 (paper A cite paper B)
                - 0 (paper A doesn't cite paper B)
    """

    # list all the papers
    nodes = list(graph.nodes())

    # list all the papers that have at least one connection
    pos_edges = list(graph.edges())

    # create a set for papers that doesn't have any connection
    neg_edges = set()

    # generate unique pairs of papers that doesn't have any connection 
    while len(neg_edges) < len(pos_edges):
        node_a, node_b = random.sample(nodes, 2)
        if not graph.has_edge(node_a, node_b):
            neg_edges.add((node_a, node_b))
            
    neg_edges = list(neg_edges)
    
    # label papers with at least one connection as 1
    pos = pd.DataFrame(pos_edges, columns=["node_a","node_b"])
    pos["Target"] = 1

    # label papers that doesn't have any connection as 0
    neg = pd.DataFrame(neg_edges, columns=["node_a", "node_b"])
    neg["Target"] = 0

    # concatenate the two datasets of positive and negative connections
    concat_df = pd.concat([pos, neg], ignore_index=True)
    
    return concat_df


def features_generation(graph: nx.DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new features based on the network and the
    corresponding dataframe, returning a panda DataFrame
    with the new informative features.
    
    Args:
        graph (nx.DiGraph): Direct graph that contain the papers as nodes
            and citations as edges

        df (pd.DataFrame): Panda DataFrame containing:
            - "node_a": Paper that cite
            - "node_b": Cited paper
            - "Target": define if paper A cite paper B
                - 1 (paper A cite paper B)
                - 0 (paper A doesn't cite paper B)
                
    Returns:
        df_feats (pd.DataFrame): Panda DataFrame containing:
            - "node_a": Paper that cite
            - "node_b": Cited paper
            - "Target": define if paper A cite paper B
                - 1 (paper A cite paper B)
                - 0 (paper A doesn't cite paper B)
            - "in_a": In-degree of the paper A (number of citations received by paper A)
            - "out_a": Out-degree of paper A (number of paper cited by A)
            - "pagerank_a": Pagerank of paper A (importance of the paper A in the network)
            - "in_b": In-degree of the paper B (number of citations received by paper B)
            - "out_b": Out-degree of paper B (number of paper cited by B)
            - "pagerank_b":Pagerank of paper B (importance of the paper B in the network)
            - "degree_ratio": Ratio between the out-degree of paper A and paper B
            - "pagerank_ratio": Ratio between Pagerank of paper A and Paper B
            - "pagerank_prod": Product of PageRank scores for paper A and paper B
            - "common_neighbors": Number of shared neighbors between paper A and paper B in the undirected graph
            - "jaccard_coeff": Jaccard similarity coefficient (normalizes common neighbors to assess relative thematic similarity)
    """

    df_feats = df.copy()

    # Compute the number of papers that cite
    # a particular paper (for all papers)
    in_degree = dict(graph.in_degree())
    
    # Compute the number of papers that a
    # particular paper cite (for all papers)
    out_degree = dict(graph.out_degree())
    
    # Compute the importance of the paper 
    # in the network (for all papers)
    pagerank = nx.pagerank(graph)
    
    # Create a undirected copy of the graph
    graph_und = graph.to_undirected()
    
    # Create a list of pairs of papers
    pairs = list(zip(df_feats["node_a"], df_feats["node_b"]))

    # Add to the DataFrame the nodes
    # features of node A
    df_feats["in_a"] = df_feats["node_a"].map(in_degree)
    df_feats["out_a"] = df_feats["node_a"].map(out_degree)
    df_feats["pagerank_a"] = df_feats["node_a"].map(pagerank)

    # Add to the DataFrame the nodes
    # features of node B
    df_feats["in_b"] = df_feats["node_b"].map(in_degree)
    df_feats["out_b"] = df_feats["node_b"].map(out_degree)
    df_feats["pagerank_b"] = df_feats["node_b"].map(pagerank)
    
    # Compute and append to the DataFrame...
    # ...the ratio of papers that the two paper cite
    df_feats["degree_ratio"] = df_feats["out_a"] / (df_feats["out_b"] + 1)
    
    # ...the prestige ratio
    df_feats["pagerank_ratio"] = df_feats["pagerank_a"] / (df_feats["pagerank_b"] + 1e-9)
    
    # ...the prestige product
    df_feats["pagerank_prod"] = df_feats["pagerank_a"] * df_feats["pagerank_b"]
    
    # ...the number of common neighbors
    df_feats["common_neighbors"] = [
    len(list(nx.common_neighbors(graph_und, u, v)))
    for u, v in pairs
    ]
    
    # ...the Jaccard coefficient
    jaccard_scores = {
    (u, v): score
    for u, v, score in nx.jaccard_coefficient(graph_und, pairs)
    }
    
    df_feats["jaccard_coeff"] = [
        jaccard_scores.get((u, v), 0)
        for u, v in pairs
    ]
    
    return df_feats



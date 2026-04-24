import pandas as pd
import networkx as nx


def network_creation(df: pd.DataFrame, art_id: str, ref_id: str, validation: str) -> nx.DiGraph:
    """
    Create from from scratch the respective network graph given the dataframe
    
    This function take the paper IDs and create connections based on the
    references between the papers
    
    Args:
        df (pd.Dataframe): dataframe containing:
            - "id": the identifier of the respective paper
            - "references": list of papers IDs
         
    Returns:
        graph (nx.DiGraph): A directed graph object where nodes are paper IDs and 
        edges represent the citations.
    """
    
    edges_df = df[df[validation]==1].copy()

    # Create connections within the graph
    graph = nx.from_pandas_edgelist(
        edges_df,
        source=art_id,
        target=ref_id,
        create_using=nx.DiGraph())
    
    all_nodes = pd.unique(df[[art_id, ref_id]].values.ravel())

    graph.add_nodes_from(all_nodes)

    return graph


def features_generation(graph: nx.DiGraph, df: pd.DataFrame, art_id: str, ref_id: str) -> pd.DataFrame:
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
            - "in_article": In-degree of the paper A (number of citations received by paper A)
            - "out_article": Out-degree of paper A (number of paper cited by A)
            - "pagerank_article": Pagerank of paper A (importance of the paper A in the network)
            - "avg_neigh_degree_article": Average neighbor degree of paper A (average degree of the papers connected to paper A)
            - "katz_cent_article":  Katz Centrality of paper A (Measure measures the influence of paper A by 
                considering both direct and indirect connections)
            - "in_ref": In-degree of the paper B (number of citations received by paper B)
            - "out_ref": Out-degree of paper B (number of paper cited by B)
            - "pagerank_ref":Pagerank of paper B (importance of the paper B in the network)
            - "eigen_cent_ref": Eigenvector centrality of paper B (importance of the papers that cite paper B)
            - "katz_cent_ref":  Katz Centrality of paper B (Measure measures the influence of paper B by 
                considering both direct and indirect connections)
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
    
    # Calculate the average of the neighborhood
    # of each node
    avg_neigh_degree = nx.average_neighbor_degree(graph, source="out", target="out")
    
    # Compute the Katz centrality for the nodes 
    # of the graph
    katz_cent = nx.katz_centrality(graph)
    
    # Compute the shortest-path betweenness 
    # centrality for nodes
    #betw_cent = nx.betweenness_centrality(graph, k=None, normalized=True)

    # Add to the DataFrame the nodes
    # features of node A and B
    nodes_names = {
        art_id:"article",
        ref_id:"ref"
        }

    for col, name in nodes_names.items():
        df_feats[f"in_{name}"] = df_feats[col].map(in_degree).fillna(0)
        df_feats[f"out_{name}"] = df_feats[col].map(out_degree).fillna(0)
        df_feats[f"pagerank_{name}"] = df_feats[col].map(pagerank).fillna(0)
        df_feats[f"avg_neigh_degree_{name}"] = df_feats[col].map(avg_neigh_degree).fillna(0)
        df_feats[f"katz_cent_{name}"] = df_feats[col].map(katz_cent).fillna(0)
        #df_feats[f"betw_cent_a"] = df_feats["node_a"].map(betw_cent).fillna(0)
    
    # Compute and append to the DataFrame...
    # ...the ratio of papers that the two paper cite
    df_feats["degree_ratio"] = df_feats[f"out_{nodes_names[art_id]}"] / (df_feats[f"out_{nodes_names[ref_id]}"] + 1)
    
    # ...the prestige ratio
    df_feats["pagerank_ratio"] = df_feats[f"pagerank_{nodes_names[art_id]}"] / (df_feats[f"pagerank_{nodes_names[ref_id]}"] + 1e-9)
    
    # ...the prestige product
    df_feats["pagerank_prod"] = df_feats[f"pagerank_{nodes_names[art_id]}"] * df_feats[f"pagerank_{nodes_names[ref_id]}"]
    
    # Create a list of pairs of papers
    pairs = list(zip(df_feats[art_id], df_feats[ref_id]))
    
    # ...the number of common neighbors
    df_feats["common_neighbors"] = [
        len(list(nx.common_neighbors(graph_und, u, v)))
        if (u in graph_und and v in graph_und) else 0
        for u, v in pairs
        ]
    
    # ...the Jaccard coefficient
    jaccard_scores = nx.jaccard_coefficient(graph_und, pairs)    
    
    df_feats["jaccard_coeff"] = [
        score for u, v, score in jaccard_scores
        ]

    return df_feats



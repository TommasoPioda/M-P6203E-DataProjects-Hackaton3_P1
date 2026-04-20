import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pyarrow.parquet as pq
from tqdm.auto import tqdm
tqdm.pandas()

os.makedirs("data", exist_ok=True)


# validation of "at least 3 years" constraint
def check_N_years(years_list, set_name, min_years=3):
    """Ensures that the provided list of years has at least the specified number of entries, printing a warning if not."""
    if len(years_list) < min_years:
        print(f"WARNING: {set_name} has only {len(years_list)} years. Adjusting...")
        return False
    return True

def get_papers_per_year(target_total, year_list, weights):
    """Calculates the number of papers to sample from each year based on the original distribution of papers and a target total."""
    # original weights but normalized to the subset of years
    subset_weights = weights.loc[year_list]
    normalized_weights = subset_weights / subset_weights.sum()
    
    # calculate count per year
    counts = (normalized_weights * target_total).round().astype(int)
    
    # adjust for rounding errors
    diff = target_total - counts.sum()
    if diff != 0:
        counts.iloc[-1] += diff
    return counts

def plot_split_distribution(cumulative_weights, years, papers_per_year, range_years= range(1971, 2025), mode=None):
    """
    - cumulative_weights: Series with cumulative weights per year.
    - years: list of 3 lists (train, val, test) with the years included in each set.
    - papers_per_year: list of 3 Series (train, val, test) with counts per year.
    - mode: Plots the data distribution across years.
        - 'weights': Cumulative distribution of weights.
        - 'papers': Number of papers selected per year (3 subplots).
        - None: Both plots.
    """
    train_years, val_years, test_years = years
    n_tr_per_year, n_val_per_year, n_test_per_year = papers_per_year
    cumulative_weights = cumulative_weights

    # helper to map colors
    def get_year_color(year):
        if year in train_years: return '#3498db' # Blue
        elif year in val_years: return '#f1c40f' # Yellow
        elif year in test_years: return '#e74c3c' # Red
        return '#bdc3c7'

    # legend elements shared by plots
    legend_elements = [
        Line2D([0], [0], color='#3498db', lw=4, label=f'Train ({min(train_years)}-{max(train_years)})'),
        Line2D([0], [0], color='#f1c40f', lw=4, label=f'Validation ({min(val_years)}-{max(val_years)})'),
        Line2D([0], [0], color='#e74c3c', lw=4, label=f'Test ({min(test_years)}-{max(test_years)})')
    ]

    # WEIGHTS PLOT
    if mode == 'weights' or mode is None:
        colors = [get_year_color(y) for y in cumulative_weights.index]
        plt.figure(figsize=(18, 6))
        plt.bar(cumulative_weights.index, cumulative_weights.values, color=colors, width=0.8)
        
        plt.title("Cumulative Distribution of Papers by Year (Chronological Split)", fontsize=15)
        plt.ylabel("Cumulative Weight", fontsize=12)
        plt.xlabel("Year", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.axvline(x=max(train_years) + 0.5, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=max(val_years) + 0.5, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # PAPERS SUBPLOTS
    if mode == 'papers' or mode is None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # setup for the 3 subplots
        sets_data = [
            (n_tr_per_year, 'Train Papers', '#3498db'),
            (n_val_per_year, 'Validation Papers', '#f1c40f'),
            (n_test_per_year, 'Test Papers', '#e74c3c')
        ]
        
        all_years = sorted(list(range_years))

        for i, (data, label, color) in enumerate(sets_data):
            # align the counts with the full range of years to keep the X-axis consistent
            counts_full = [data.get(y, 0) for y in all_years]
            axes[i].bar(all_years, counts_full, color=color, width=0.8)
            axes[i].set_ylabel("Count", fontsize=10)
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].grid(axis='y', linestyle=':', alpha=0.6)
            
        plt.xlabel("Year", fontsize=12)
        plt.suptitle("Number of Selected Papers per Year", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def sample_from_year(year, n_needed, cleaned_data_path, random_state=42):
    """Loads the parquet file for the specified year and samples the requested number of papers, ensuring reproducibility with a random state."""
    if n_needed <= 0: return pd.DataFrame()
    path = f"{cleaned_data_path}/year_{year}.0.parquet"
    df = pd.read_parquet(path)
    # n=min(n_needed, len(df)) ensures we don't crash if a year is smaller than expected
    return df.sample(n=min(n_needed, len(df)), random_state=random_state)
        

def analyze_set_connectivity(df, set_name="Dataset", show_graph=False, show_distribution=False, verbose=True):
    """
    Explodes references, creates a network graph, and calculates comprehensive 
    connectivity statistics including cluster size distributions.
    """
    print(f"\n--- Starting Connectivity Analysis for: {set_name} ---")

    # 1. References Explosion
    # converting NaNs/Floats to empty lists
    print(f"[{set_name}] Cleaning and exploding references...")
    df_clean = df.copy()
    df_clean['references'] = df_clean['references'].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else []
    )

    # Explode the dataframe to get (citing_id, cited_id) pairs
    exploded = df_clean[['id', 'references']].explode('references').dropna(subset=['references'])

    # 2. Graph Construction
    print(f"[{set_name}] Building the network graph...")
    G = nx.Graph() # Undirected graph to find interconnected groups (clusters)
    
    # all unique paper IDs as nodes
    all_ids = set(df['id'].unique())
    G.add_nodes_from(all_ids)

    # filter for internal edges only (both citing and cited must be in this set)
    internal_edges = exploded[exploded['references'].isin(all_ids)]
    
    # add edges to the graph
    edges = list(zip(internal_edges['id'], internal_edges['references']))
    for u, v in tqdm(edges, desc=f"Adding edges for {set_name}"):
        G.add_edge(u, v)

    # 3. Connected Components (Clusters) Identification
    print(f"[{set_name}] Identifying connected components...")
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    num_components = len(components)
    total_nodes = len(G)

    if total_nodes == 0:
        print(f"[{set_name}] Error: The dataset is empty.")
        return None

    # 4. Statistics Calculation
    component_sizes = [len(c) for c in components]
    size_counts = Counter(component_sizes) # counts how many clusters have size 1, 2, 5, etc.
    
    largest_cc_size = component_sizes[0]
    smallest_cc_size = component_sizes[-1]

    largest_pct = (largest_cc_size / total_nodes) * 100
    smallest_pct = (smallest_cc_size / total_nodes) * 100

    if verbose: 
        print('-' * 30)
        print(f"\nResults for {set_name}")
        print(f"\tTotal Papers (Nodes): {total_nodes}")
        print(f"\tTotal Clusters (Connected Groups): {num_components}")
        print(f"\tLargest Cluster Size: {largest_cc_size} papers ({largest_pct:.2f}%)")
        print(f"\tSmallest Cluster Size: {smallest_cc_size} papers ({smallest_pct:.2f}%)")
        
        # cluster Size Distribution (Value Counts)
        print(f"\n\tCluster Size Distribution (Frequency):")
        # sort by cluster size for better readability
        for size in sorted(size_counts.keys()):
            count = size_counts[size]
            print(f"\t\t- Clusters of size {size}: {count}")

    # 5. Visualization of the Graph and Cluster Size Distribution
    # graph viz
    if show_graph:
        print(f"[{set_name}] Generating visualization...")
        plt.figure(figsize=(12, 8))
        if total_nodes > 1000:
            print(f"  [Note] Node count > 1000. Visualizing only the largest component for clarity.")
            subgraph = G.subgraph(components[0])
            pos = nx.spring_layout(subgraph, k=0.15, seed=42)
            nx.draw(subgraph, pos, node_size=20, node_color="skyblue", edge_color="silver", alpha=0.6)
        else:
            pos = nx.spring_layout(G, k=0.15, seed=42)
            nx.draw(G, pos, node_size=50, node_color="orange", edge_color="gray", alpha=0.5)
        
        plt.title(f"Network Connectivity - {set_name}")
        plt.show()

    # size distribution viz
    if show_distribution:
        print(f"[{set_name}] Plotting cluster size distribution...")
        plt.figure(figsize=(10, 6))

        # order the size_counts by cluster size
        sorted_size_items = sorted(size_counts.items(), key=lambda x: x[0])

        if len(sorted_size_items) > 30:
            print(f"  [Note] Too many unique sizes ({len(sorted_size_items)}). Showing only the most frequent or largest.")
            display_items = sorted_size_items[:20] + [sorted_size_items[-1]]
        else:
            display_items = sorted_size_items

        labels = [str(item[0]) for item in display_items]
        counts = [item[1] for item in display_items]
        bars = plt.bar(labels, counts, color='royalblue', alpha=0.8, edgecolor='black')

        plt.yscale('log')

        plt.xlabel("Cluster Size (Categorical Labels)", fontsize=12)
        plt.ylabel("Number of Clusters (Log Scale)", fontsize=12)
        plt.title(f"Cluster Size Distribution - {set_name}", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        # text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                     va='bottom', ha='center', fontsize=9, fontweight='bold')

        plt.show()

    return {
        "num_clusters": num_components,
        "largest_pct": largest_pct,
        "size_distribution": dict(size_counts)
    }
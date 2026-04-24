cured by [Bernacchia Alessia](https://github.com/AlessiaBernacchia), [Pioda Tommaso](https://github.com/Thetommigun432), [Villani Giacomo](https://github.com/DownToTheGround)

# Project Report

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)
2. [Cleaning Pipeline](#2-cleaning-pipeline)
3. [Feature Selection](#3-feature-selection)
4. [Splitting into Train, Validation and Test set](#4-splitting-into-train-validation-and-test-set)
5. [Remaining Data Characteristics & Considerations](#5-remaining-data-characteristics--considerations)
6. [Feature Engineering](#6-feature-engineering)
7. [Models](#7-models)
8. [Comparison](#8-comparison)
9. [Interpretability](#9-interpretability)
10. [Conclusions](#10-conclusions)
---

## 1. Dataset Overview
### 1.1 Description
The dataset is a comprehensive collection of scientific papers sourced from the DBLP (Computer Science Bibliography). 

It is designed for tasks involving data exploration, citation prediction, and classification within the academic domain.

| Property | Details |
|----------|---------|
| **Source** | [DBLP (Computer Science Bibliography)](https://opendata.aminer.cn/dataset/DBLP-Citation-network-V18.zip) |
| **Total Records** | 6.7+ million academic papers |
| **Original Format** | JSONL (JSON Lines) |
| **Processing Format** | Apache Parquet |
| **Time Period** | 1800s to 2024 |

### Why Parquet?
- **Columnar Storage**: parquet stores data by column rather than by row, enabling selective column retrieval
- **Query Performance**: columnar datasets significantly outperform row-based formats (CSV, JSON) for analytical queries
- **Compression**: better compression ratios reduce storage requirements
- **Schema Preservation**: maintains data types and nested structures

### 1.2 Key Fields
![first line of the dataset](src/exp_raw%20lines.png)
The dataset contains 18 primary fields organized into three categories:

**Core Metadata:**
- `id` (string): unique paper identifier
- `title` (string): research paper title
- `year` (integer): publication year
- `lang` (string): detected document language
- `doi` (string): Digital Object Identifier link (mix alpha and num chars)

**Publication Details:**
- `doc_type` (categorical string): publication type (conference, journal)
- `venue` (string): publishing venue name
- `abstract` (string): paper summary
- `authors` (array[dict]): author information (name, ID, organization)

**Citation & Reference Data:**
- `references` (array[string]): cited paper IDs
- `n_citation` (integer): citation count
- `keywords` (array[string]): relevant tags or index terms

**Bibliographic Information:**
- `volume` (integer): volume number of publication
- `issue` (integer): issue number of publication
- `page_start`, `page_end` (integers): starting/ending page number in publication
- `isbn`, `issn` (strings): Internationa Standard Book/Serial Number (identifiers with also alpha values)
- `url` (array[string]): external resource links

### 1.3 Categorize Fields
Thanks to this analysis we identified the categories:
- ***numeric columns***: 'year', 'n_citation', 'page_start', 'page_end', 'volume', 'issue'
- ***string columns***: 'id', 'title', 'lang', 'doc_type', 'venue', 'issn', 'isbn', 'doi', 'abstract'
- ***arrays of strings***: 'references', 'keywords', 'url'
- ***arrays of dictionaries*** (*specific structure*): 'authors'

### 1.4 Exploration
#### Global-Level Analysis
![global raw data description](src/exp_global%20raw%20descr.png)
![publications per year histogram](src/exp_raw%20pubs4year.png)
> There are 48 years in which there are less than 50 publications...
![nans per feature histogram](src/exp_raw%20nan.png)

#### Field-Level Analysis Summary
Initial field-level inspection identified key data quality issues that guided subsequent cleaning:

| Field | Key Findings | Impact |
|-------|--------------|--------|
| Authors | 0.2% NaN, 1.8% empty lists, 45K+ unique authors | Validated structure integrity |
| Keywords | 8.5% NaN, 2.1% empty arrays, 120K+ unique terms | Acceptable missing rate |
| Venue | 3.2% NaN, 8,500+ unique venues | Identified mismatch potential |
| Doc_type | 0% NaN, mixed case variants present | Standardization needed |
| Year | 0% NaN, range 1800–2027 with invalid entries | Temporal filtering required |

These findings informed the cleaning strategy outlined in Section 2.1 .
---

## 2 Cleaning Pipeline
### 2.1 Data Type Validation & Normalization
Normalize all the features based on the category type assigned.
#### String Column Processing
- Assign _string_ type
- Replaced NULL placeholders (`None`, `nan`, `N/A`, `-`, `/`, `?`, `null`) with `np.nan`

#### Numeric Column Processing
- Coerced to numeric type with error handling (`errors='coerce'`)
- Converted negative values to 0
- Applied domain constraints:
    - **Year**: Removed entries where `year <= 1800` or `year > current_year`
    - **Page numbers**: Ensured `page_start ≤ page_end`
    - **Citation count**: Removed negative values

#### Categorical Processing
- Standardized `doc_type` to lowercase
- Restricted `doc_type` to valid categories: `['conference', 'journal']`
- Marked non-conforming entries as NaN for imputation

#### List-Based Columns
- Validated array structures (must be `np.ndarray` with string elements)
- Converted invalid structures to NaN
- **Special case for references**: Preserved empty arrays (`[]`) as valid indicators of "no references"

### 2.2 Authors Validation
#### Authors Registry Construction
To validate, fill the gaps in the authors information, it's necessary to build a comprehensive registry of unique authors, permitting us to access information about a specific author.

Extract $\rightarrow$ Aggregate $\rightarrow$ Clean $\rightarrow$ Enrich $\rightarrow$ (Save)

**Extract**: capture for each author
- name used
- id (when available)
- organizations associated
- year of the paper (needed to make organisation history)
- language of the publication
- keywords treated

**Aggregate**: merge duplicate authors across papers
- list of multiple name variants
- track organisation history list[(org, year)]
- languages of publications

**Cleaning**: 
- removed invalid names (non-alphabetic)
- deduplicate organisation-year pairs, keywords and languages
- remove organisation-year pairs with a missing value

**Enrichment**:
- assign _official name_ (based on most frequent name)
- assign _id_ using registry cross-references
- index organization by year

Final Author Registry Schema:
| Column | Type | Purpose |
|--------|------|---------|
| `id` | string | Primary key (author identifier) |
| `official_name` | string | Canonical author name |
| `name` | array[string] | Historical name variations |
| `org_year` | array[dict] | Organization timeline (`{org, year}`) |
| `keywords` | array[string] | Associated research topics |
| `lang` | array[string] | Publication languages |

#### Gap Filling
Three-step gap filling process:
1. **ID Assignment**: authors without ID assigned using official name registry (map _name_ $\rightarrow$ _official name_)
2. **Name Completion**: missing names filled from ID-official name mappings
3. **Organization Inference**: missing affiliations inferred from `(author_id, publication_year)` index

**Example workflow:**
```
Input:  {name: null, id: "auth_12345", org: null, year: 2020}
Step 1: Name lookup → official_name: "John Smith"
Step 2: Org lookup → (12345, 2020) → "MIT"
Output: {name: "John Smith", id: "auth_12345", org: "MIT"}
```

### 2.3 Venue Validation
Implemented heuristic-based corrections:

| Condition | Action | Count |
|-----------|--------|-------|
| "Conference" in venue, doc_type="journal" | Set to "conference" | ~84,000 |
| "Journal" in venue, doc_type="conference" | Set to "journal" | ~20,000 |
| Venue indicates conference, doc_type=null | Set to "conference" | ~400 |
| Venue indicates journal, doc_type=null | Set to "journal" | ~25 |

**Total records corrected**: ~100,500

### 2.4 References Validation
Each reference was validated against three criteria:
- **valid ID**: cannot cite an ID not present in registry
- **temporal consistency**: cannot cite a future paper not written jet (reference year $<=$ paper year)
- **not null**: reference should be a valid identifier

Example:
```
Paper ID: 5390877920f70186a0d2ce7f, Year: 1984
Here the complete references of the paper: ['5390879d20f70186a0d43d74', '5390a1e620f70186a0e59c05']
    Reference ID: 5390879d20f70186a0d43d74, Year: 1984.0	is valid? True
    Reference ID: 5390a1e620f70186a0e59c05, Year: 1985.0	is valid? False
```
---

## 3. Feature Selection
### 3.1 Exploration Analysis
![publications per year histogram](src/exp_clean%20pubs4year.png)

> Not significant years are deleted. Since we've a huge number of publications we can consider to select only a part of them, but we need to consider the year representation.

![nans per feature histogram](src/exp_clean%20nan.png)

> The number of missing values increases after cleaning because empty arrays and invalid values are converted to NaNs. In particular, the features isbn, issue, volume, and issn show a high percentage of missing values (50% or more).

### 3.2 Removal of Non-Meaningful Features

After cleaning, several features were identified as non-essential due to high missing value percentages or limited analytical value:

**Removed features:**
- **isbn, issn**: >50% missing values; not critical for citation analysis
- **issue, volume**: >50% missing values; insufficient for predictive modeling
- **page_start, page_end**: Replaced with derived feature `n_pages` to preserve information about paper length

**New derived feature:**
- **n_pages**: Calculated as `page_end - page_start + 1`, representing paper length which may correlate with citation count

### 3.3 Final Feature Set

| Category | Features |
|----------|----------|
| **Core Identifiers** | id, title, year, lang |
| **Publication Info** | doc_type, venue, n_pages, abstract |
| **Author Data** | authors (array of {id, name, org}) |
| **Citation & References** | n_citation, references (array), keywords (array) |
| **URLs** | url (array) |

---

## 4. Splitting into Train, Validation and Test set
### 4.1 Overview of Splitting Strategy
The dataset is split into train, validation, and test sets using a year-weighted, chronological approach to preserve temporal distribution and avoid bias. Only papers from 1971–2024 are considered, as earlier years have sparse data (<50 publications/year), and later years lack sufficient records.

#### Key Steps:
1. **Year Range Selection**: Filter papers to 1971–2024 based on publication density.
2. **Weight Calculation**: Compute weights proportional to annual publication counts to maintain real-world distribution.
3. **Chronological Assignment**: Divide years into contiguous blocks (train: earliest, validation: middle, test: latest), ensuring at least 10 years per set for representativeness.
4. **Proportional Sampling**: For each year, sample papers proportionally to its weight, targeting 500,000 train, 150,000 validation, and 150,000 test papers.
5. **Random Sampling & Shuffling**: Sample without replacement within years, then globally shuffle each set.
6. **Connectivity Verification**: Analyze graph connectivity to ensure sets are well-linked internally.

> In order to obtain an acceptable size of the biggest connected cluster, we tried different combinations of random seeds and minimum years sampled for set. We obtained the best result with `random_seed=21` and `min_years=10`, maybe we could obtain better result with higher `min_years`, but put more could _leave training data with obsolete data_, then we decided to maintain this. The training set has the largest cluster of connected nodes that cover ~67% of the entire set, validation has ~17% and test set ~20.

This method prevents over-representation of sparse years and under-sampling of abundant ones, ensuring realistic temporal balance.

**Example:**
- If year 2000 has 2% of all papers $\rightarrow$ allocate 2% of train samples from 2000
- Prevents early years (sparse data) from being over-represented
- Prevents recent years (abundant data) from dominating

### 4.2 Final Split Sizes
![global zear of publication distribution divided by sets](src/split_distr%20years.png)
```
global year range selected: from 1971 to 2014
random seed: 21
min number of years per set: 10
```

| Split | Years Covered | % of samples in biggest connected cluster | Target Size | Actual Size |
|-------|-------------|-------------|-------------|-------------|
| Train | 1971 - 2004 | 67.64% | 500,000 | 500,000 |
| Validation | 2005 - 2014 | 17.00% | 150,000 | 150,000 |
| Test | 2015 - 2024 | 19.66% | 150,000 | 150,000 |

> The validation and test set has a smaller % of samples on the biggest connected cluster makes sense because they cover a smaller temporal range. Usually, to start to be cited, the paper should be published for some years in order to be known.
---

## 5. Remaining Data Characteristics & Considerations

### 5.1 Data Quality Issues & Resolutions

| Issue | Root Cause | Resolution |
|-------|-----------|------------|
| **Future References** | Temporal inconsistencies in citation data | Removed references with `ref_year > paper_year` |
| **Invalid Years** | Data entry errors, parsing anomalies | Filtered `year <= 1800` or `year > current_year` |
| **Venue-DocType Mismatch** | Inconsistent metadata classification | Applied heuristic corrections based on venue keywords |
| **Missing Author IDs** | Incomplete metadata in source | Cross-referenced with authors registry |
| **Missing Affiliations** | Sparse organization data | Inferred from author-year history index |
| **Empty Arrays** | Absent data vs. intentional nulls | Converted to NaN for consistent handling |
| **Doc_type Inconsistencies** | Multiple case variants, invalid categories | Standardized to lowercase `['conference', 'journal']` |

### 5.2 Known Limitations

**Sparse Temporal Coverage:**
- Years before 1971 have insufficient publication density (<50 papers/year)
- Stratified sampling focuses on 1971–2024; earlier periods underrepresented

**Missing Values Persist:**
- `abstract`: ~25% missing (limit NLP applications on full text)
- `keywords`: ~8.5% missing (affects topic-based analysis)
- `url`: ~35% missing (external validation limited)
- `n_pages`: ~40% missing (derived from missing page boundaries)

**Author Ambiguity:**
- Author name variations still exist despite standardization
- Cross-institutional author disambiguation remains imperfect
- ~2% of authors still lack validated IDs

**Venue Information:**
- ~3% missing venue entries (especially for older papers)
- Venue name variations not fully normalized (e.g., "ACM Conference" vs. "ACM SIGMOD")
- Smaller conferences underrepresented

## 6. Feature Engineering

### 6.3 Graph Feature

**Network Construction**

- The network is built as a directed graph where each edge represents a validated citation (`Target = 1`) pointing from the citing article to the referred paper.
- To ensure data consistency and prevent `NodeNotFound` errors during inference, all papers present in the dataset—including isolated nodes without active citations—have been explicitly added to the graph.

**Feature Categorization**

The extracted graph features are divided into three main categories based on their structural depth:

- `Node Features (Individual Importance)`: These describe the role and authority of each individual article within the network:
    * **in_***: *In-degree*; the number of citations received (a measure of popularity/prestige).
    * **out_***: *Out-degree*; the number of references made (a measure of bibliographic breadth).
    * **pagerank_***: A centrality score defining the relative importance of the paper based on the quality of its incoming citations.
    * **avg_neigh_degree_***: The average degree of a node's neighbors, helping to identify nodes connected to major hubs or isolated clusters.
    * **katz_cent_***: An influence measure that considers both direct and long-range indirect connections.

- `Neighborhood Features (Local Context)`: These analyze the local overlap between the article and the reference:
    * **common_neighbors**: The absolute count of shared neighbors in the undirected version of the graph, indicating a common scientific foundation.



- `Relational Features (Pairwise Interaction)`: These capture the hierarchical dynamics and specific structural similarity between the pair of nodes:
    * **degree_ratio**: The ratio between the out-degree of the article and the reference, used to balance the activity levels of the two nodes.
    * **pagerank_ratio**: The disparity in importance between the article and the reference; it identifies typical patterns, such as new papers citing "classics."
    * **pagerank_prod**: The product of the importance of both nodes, highlighting connections between two pillar nodes of the network.
    * **jaccard_coeff**: A structural similarity coefficient that normalizes the number of shared neighbors relative to the total size of their combined neighborhoods:

$$J(A, B) = \frac{|N(A) \cap N(B)|}{|N(A) \cup N(B)|}$$

## 7. Models

## 8. Comparison

## 9. Interpretability

## 10. Conclusions
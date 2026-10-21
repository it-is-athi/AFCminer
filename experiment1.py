"""
This file contains Experiment 1 described in the paper:
1. Binary-valued attribute experiment using gender.

Subsets are pre-generated upfront according to SUBSET_SIZES in experiment_utils.py
and reused for all visualization tables and mining algorithms.
"""

from collections import Counter
from time import perf_counter
from experiment_utils import (
    TOP_NODE_COUNT,
    DATASET_FILE,
    DATASET_FOLDER,
    SUBSET_SIZES,
)
from sparse_preprocess import load_facebook100_data, split_preprocessed_data
from fc_miner import FairCliqueMiner
from bk import BKMiner

DEFAULT_K = 1
DEFAULT_DELTA = 2


def pregenerate_subsets():
    """Generates and stores all subset datasets upfront for SUBSET_SIZES."""
    generated_subsets = {}
    print("\n[Preprocessing] Pre-generating all graph subsets upfront...")
    for size in SUBSET_SIZES:
        nodes, attribute_columns, combined_data = load_facebook100_data(
            mat_filename=DATASET_FILE,
            folder_name=DATASET_FOLDER,
            max_nodes=size,
            attribute_type="gender",
            granularity=2
        )
        generated_subsets[size] = (nodes, attribute_columns, combined_data)
    print(f"[Preprocessing] Completed loading {len(generated_subsets)} subsets.\n")
    return generated_subsets


def print_table_ii(generated_subsets):
    """Print friendship statistics using the pre-generated subset datasets."""
    print("\nTABLE II")
    print("DATASETS I. FRIENDSHIP OF USERS")
    print("================================")
    print(f"\n{'Sub-Dataset':<14} {'Node':>6} {'Edge':>8} {'d_avg':>8} {'d_max':>8}")
    print("-" * 48)

    for index, size in enumerate(SUBSET_SIZES, start=1):
        nodes, _, combined_data = generated_subsets[size]
        edges, _ = split_preprocessed_data(nodes, combined_data)

        node_count = len(nodes)
        edge_count = len(edges)
        average_degree = (2 * edge_count) / node_count if node_count > 0 else 0
        degrees = {node: 0 for node in nodes}
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        maximum_degree = max(degrees.values(), default=0)

        print(
            f"SubSet {index:<7} {node_count:>6} {edge_count:>8} "
            f"{average_degree:>8.2f} {maximum_degree:>8}"
        )

    print("-" * 48)

def run_experiment_1(generated_subsets, k=DEFAULT_K, delta=DEFAULT_DELTA):
    wfc_result_rows = []
    runtime_rows = []
    size_breakdown_data = []

    print(f"\nRunning Experiment 1 with Parameters: k={k}, delta={delta}")

    for index, size in enumerate(SUBSET_SIZES, start=1):
        nodes, attribute_columns, combined_data = generated_subsets[size]
        print(f"\nProcessing SubSet {index} ({size} nodes)...", flush=True)

        V = nodes
        node_attribute_set = attribute_columns
        R = combined_data

        # 1. Run BKMiner with timing
        print("  Running BKMiner...", flush=True)
        start_time = perf_counter()
        bk_out = BKMiner(V, node_attribute_set, R, k=k, delta=delta)
        bk_time_ms = (perf_counter() - start_time) * 1000

        if len(bk_out) == 6:
            bk_afc, bk_wfc, bk_afc_max, bk_wfc_max, bk_count_afc, bk_count_wfc = bk_out
        else:
            bk_afc, bk_wfc = bk_out[0], bk_out[1]

        # 2. Run FairCliqueMiner with timing
        print("  Running FairCliqueMiner...", flush=True)
        start_time = perf_counter()
        fair_clique_out = FairCliqueMiner(
            V,
            node_attribute_set,
            R,
            k=k,
            delta=delta
        )
        fair_clique_time_ms = (
            perf_counter() - start_time
        ) * 1000

        (
            absolute_fair_cliques,
            weak_fair_cliques,
            largest_absolute_fair_size,
            largest_weak_fair_size,
            absolute_fair_count_by_size,
            weak_fair_count_by_size
        ) = fair_clique_out

        runtime_rows.append(
            (
                index,
                size,
                bk_time_ms,
                fair_clique_time_ms
            )
        )

        wfc_result_rows.append((
            index,
            len(absolute_fair_cliques),
            len(weak_fair_cliques),
            largest_absolute_fair_size,
            largest_weak_fair_size
        ))

        size_breakdown_data.append((
            index,
            size,
            absolute_fair_count_by_size,
            weak_fair_count_by_size
        ))

    # --- TABLE V: RUNNING TIME COMPARISON (ms) ---
    print("\nTABLE V")
    print("EXPERIMENT-1. RUNNING TIME COMPARISON (ms)")
    print("==========================================================")
    print(f"{'Dataset':<10} {'Nodes':>8} {'BK (ms)':>16} {'AFCMiner (ms)':>18}")
    print("-" * 56)
    for idx, sz, b_time, a_time in runtime_rows:
        print(f"SubSet {idx:<4} {sz:>8} {b_time:>16.2f} {a_time:>18.2f}")
    print("-" * 56)

    # --- SUMMARY: AFC VS WFC OVERALL CLIQUE COUNT & MAX SIZE ---
    print("\nWFC VS AFC COMPARISON TABLE")
    print("==========================================================================================")
    print(f"{'Dataset':<10} {'Total AFC':>12} {'Total WFC':>12} {'Max Size (AFC)':>16} {'Max Size (WFC)':>16} {'Extra Cliques (WFC-AFC)':>24}")
    print("-" * 94)
    for idx, afc_cnt, wfc_cnt, afc_max, wfc_max in wfc_result_rows:
        diff = wfc_cnt - afc_cnt
        print(f"SubSet {idx:<4} {afc_cnt:>12} {wfc_cnt:>12} {afc_max:>16} {wfc_max:>16} {diff:>24}")
    print("-" * 94)

    # --- DETAILED PER-CLIQUE-SIZE BREAKDOWN ---
    print("\nPER-SIZE CLIQUE DISTRIBUTION COMPARISON")
    print("==========================================================================================")
    
    for idx, size, count_afc, count_wfc in size_breakdown_data:
        all_sizes = sorted(set(count_afc.keys()) | set(count_wfc.keys()))
        print(f"\n--- SubSet {idx} ({size} Nodes) ---")
        print(f"{'Clique Size (|C|)':<20} {'AFC Count':>15} {'WFC Count':>15}")
        print("-" * 54)
        
        for c_size in all_sizes:
            a_num = count_afc.get(c_size, 0)
            w_num = count_wfc.get(c_size, 0)
            print(f"{c_size:<20} {a_num:>15} {w_num:>15}")
        print("-" * 54)

def load_attributes_for_table_iii():
    """Load attribute category distributions for Table III."""
    attribute_settings = [
        ("Status", "status", 2),
        ("Gender", "gender", 2),
        ("Major", "major", 5),
        ("Year", "year", 4),
        ("Dorm", "dorm", 5),
    ]
    attribute_groups = []

    for dataset_name, attribute_type, granularity in attribute_settings:
        nodes, _, combined_data = load_facebook100_data(
            mat_filename=DATASET_FILE,
            folder_name=DATASET_FOLDER,
            max_nodes=TOP_NODE_COUNT,
            attribute_type=attribute_type,
            granularity=granularity,
        )
        _, node_attributes = split_preprocessed_data(nodes, combined_data)
        attribute_groups.append((dataset_name, nodes, node_attributes))

    return attribute_groups


def print_table_iii():
    """Print category names, tags, and counts for Table III."""
    attribute_groups = load_attributes_for_table_iii()

    print("\nTABLE III")
    print("DATASETS II. CATEGORIES OF DIFFERENT ATTRIBUTES")
    print("===============================================")
    print(f"\n{'Dataset':<12} {'Category':<18} {'Tag':<5} {'Number':>8}")
    print("-" * 48)

    for dataset_name, nodes, node_attributes in attribute_groups:
        counts = Counter(node_attributes.values())
        for tag, (category, count) in enumerate(
            sorted(counts.items()), start=1
        ):
            print(
                f"{dataset_name:<16} {category:<18} {tag:<5} "
                f"{count:>8}"
            )
        print("-" * 48)


if __name__ == "__main__":
    # 1. Pre-generate subsets upfront for SUBSET_SIZES
    subsets_data = pregenerate_subsets()

    # 2. Print Table II using pre-generated datasets
    print_table_ii(subsets_data)

    # 3. Print Table III (Independent)
    print_table_iii()

    # 4. Run AFCMiner and BKMiner using pre-generated datasets
    run_experiment_1(subsets_data)
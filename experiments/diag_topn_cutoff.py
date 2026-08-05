"""
Diagnostic: What percentage of correct SIRETs are being cut off by the Stage 1 Top-N=20 limit?

Uses the decider training samples (samples_v5_decider.parquet).
The parquet stores one row per (query, candidate) pair with a binary label.
We derive the rank of each candidate within its query by sorting by decider score
descending, then check where the positive (correct) candidate falls.

Ground truth label is used ONLY for evaluation — never as a model input feature.
"""

import pandas as pd
from pathlib import Path

samples_path = Path("data/samples_v5_decider.parquet")
print(f"Loading {samples_path}...")
df = pd.read_parquet(samples_path)
print(f"Total rows: {len(df):,}")
print(f"Unique queries: {df['query_id'].nunique():,}")
print()

# Derive within-query rank using the best available score column
# (name_jaro_max is the strongest single signal for ranking position)
score_col = "name_jaro_max"
print(f"Computing within-query rank using '{score_col}'...")
df = df.sort_values(["query_id", score_col], ascending=[True, False])
df["rank_within_query"] = df.groupby("query_id").cumcount() + 1

# Positive samples = correct match for each query
positives = df[df["label"] == 1].copy()
print(f"Total positive samples (correct matches): {len(positives):,}")
print()

print("Rank distribution of correct matches (top 30):")
rank_dist = positives["rank_within_query"].value_counts().sort_index()
print(rank_dist.head(30).to_string())
print()

# Key statistics
total_pos = len(positives)
in_top_20 = positives[positives["rank_within_query"] <= 20]
in_top_50 = positives[positives["rank_within_query"] <= 50]
beyond_20 = positives[positives["rank_within_query"] > 20]
beyond_50 = positives[positives["rank_within_query"] > 50]

print(f"\n{'='*55}")
print(f"Correct matches in Top 1:   {len(positives[positives['rank_within_query']==1]):,} / {total_pos:,}  ({100*len(positives[positives['rank_within_query']==1])/total_pos:.2f}%)")
print(f"Correct matches in Top 5:   {len(positives[positives['rank_within_query']<=5]):,} / {total_pos:,}  ({100*len(positives[positives['rank_within_query']<=5])/total_pos:.2f}%)")
print(f"Correct matches in Top 20:  {len(in_top_20):,} / {total_pos:,}  ({100*len(in_top_20)/total_pos:.2f}%)")
print(f"Correct matches in Top 50:  {len(in_top_50):,} / {total_pos:,}  ({100*len(in_top_50)/total_pos:.2f}%)")
print(f"---")
print(f"LOST by Top-20 cut (rank 21-50):    {len(beyond_20)-len(beyond_50):,} ({100*(len(beyond_20)-len(beyond_50))/total_pos:.2f}%)")
print(f"LOST by Top-50 cut (rank 51+):      {len(beyond_50):,} ({100*len(beyond_50)/total_pos:.2f}%)")
print(f"{'='*55}")

# Distribution of pool sizes per query (how many candidates per query)
pool_sizes = df.groupby("query_id").size()
print(f"\nCandidate pool size stats (per query):")
print(f"  Min: {pool_sizes.min()}")
print(f"  Median: {pool_sizes.median():.0f}")
print(f"  Mean: {pool_sizes.mean():.1f}")
print(f"  Max: {pool_sizes.max()}")
print(f"  Queries with >20 candidates: {(pool_sizes>20).sum():,} / {len(pool_sizes):,}")

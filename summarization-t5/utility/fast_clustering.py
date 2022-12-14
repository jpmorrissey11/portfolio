import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import argparse


def parse_args() -> argparse.Namespace:
    # create the parser
    parser = argparse.ArgumentParser()
    # parser = Experiment.add_experiment_specific_args(parser)
    parser.add_argument(
        "--subset_size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--min_community_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument("--data_dir", type=str, help="directory of data to cluster")
    parser.add_argument("--dataset_name", type=str, help="name of dataset to cluster")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save clustered data",
    )

    # parse the arguments from the command line
    args = parser.parse_args()
    return args


args = parse_args()


model = SentenceTransformer("all-MiniLM-L6-v2")

df, corpus_sentences = prepare_data(
    args.data_dir, args.dataset_name, "description", args.subset_size
)
breakpoint()
print("Encode the corpus, this might take a while")
corpus_embeddings = model.encode(
    corpus_sentences, batch_size=32, show_progress_bar=True, convert_to_tensor=True
)

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
min_community_size = 2
threshold = 0.7
clusters = util.community_detection(
    corpus_embeddings, min_community_size=min_community_size, threshold=threshold
)

num_clusters = len(clusters)
cluster_sizes = [len(c) for c in clusters]
avg_size = np.mean(cluster_sizes)
num_clustered_sentences = np.sum(cluster_sizes)
num_unclustered_sentences = df.shape[0] - np.sum(cluster_sizes)

cluster_metadata = {
    "num_sentences": df.shape[0],
    "num_clustered_sentences": num_clustered_sentences,
    "num_unclustered_sentences": num_unclustered_sentences,
    "num_clusters": num_clusters,
    "cluster_sizes": cluster_sizes,
    "avg_size": avg_size,
}

# assign a cluster value for dataframe cells that were clustered
for i, cluster in enumerate(clusters):
    for sentence_id in cluster:
        df.at[sentence_id, "cluster"] = i

clustered_df = df[["description", "self_serviced", "cluster"]].copy()

clustered_df.to_csv(os.path.join(args.save_dir, "clustered_data.csv"), index=False)

for key, value in cluster_metadata.items():
    print(f"{key}: {value}")

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i + 1, len(cluster)))
    for sentence_id in cluster[0:2]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-2:]:
        print("\t", corpus_sentences[sentence_id])

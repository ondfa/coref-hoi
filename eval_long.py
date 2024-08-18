import sys
import wandb
import torch
import numpy as np

from run import Runner

def evaluate(runner):
    data = runner.data.tensor_samples["dev"]
    num_segments = [1, 4, 8]
    metrics = {}
    metrics["total_segments"] = sum([ex[1][0].shape[0] for ex in data])
    for num_s in num_segments:
        _, cross_corefs = runner.find_cross_segment_coreference(data, num_s)
        num_cross_corefs = sum([torch.sum(cross_coref).item() for cross_coref in cross_corefs])
        num_corefs = 0
        num_mentions = 0
        for example in data:
            cluster_ids = example[1][-1]
            same_cluster = torch.triu(torch.unsqueeze(cluster_ids, dim=0) == torch.unsqueeze(cluster_ids, dim=1), diagonal=1)
            num_corefs += torch.sum(same_cluster).item()
            num_mentions += same_cluster.shape[0]
        # num_corefs = sum([cross_coref.shape[0] for cross_coref in cross_corefs])
        metrics[f"num_cross_{num_s}_segments_corefs"] = num_cross_corefs
        metrics[f"portion_cross_{num_s}_segments_corefs"] = num_cross_corefs / num_corefs
        _, nearest_cross_corefs = runner.find_cross_segment_coreference(data, num_s, nearest_only=True)
        num_nearest_cross_corefs = sum([torch.sum(cross_coref).item() for cross_coref in nearest_cross_corefs])
        metrics[f"num_nearest_cross_{num_s}_segments_corefs"] = num_nearest_cross_corefs
        metrics[f"portion_nearest_cross_{num_s}_segments_corefs"] = num_nearest_cross_corefs / num_mentions
        segments_over_max_len = sum([max(0, ex[1][0].shape[0] - num_s) for ex in data])
        total_sements = sum([ex[1][0].shape[0] for ex in data])
        over_max_portion = segments_over_max_len / total_sements
        metrics[f"num_segments_over_{num_s}"] = segments_over_max_len
        metrics[f"portion_sements_over_{num_s}"] = over_max_portion
    wandb.log(metrics)
    print(metrics)

if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = Runner(config_name, gpu_id)
    evaluate(runner)

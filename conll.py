import re
import tempfile
import subprocess
import operator
import collections
import logging

import udapi.core.coref
from udapi.core.coref import CorefCluster, span_to_nodes

logger = logging.getLogger(__name__)

BEGIN_DOCUMENT_REGEX = re.compile(r"#begin document \((.*)\); part (\d+)")  # First line at each document
BEGIN_DOCUMENT_REGEX_COREFUD = re.compile(r"# newdoc id = (.*)")  # First line at each document
COREF_RESULTS_REGEX = re.compile(r".*Coreference: Recall: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tPrecision: \([0-9.]+ / [0-9.]+\) ([0-9.]+)%\tF1: ([0-9.]+)%.*", re.DOTALL)


def get_doc_key(doc_id, part):
    return "{}_{}".format(doc_id, int(part))


def output_conll(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k,v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k,v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX, line)
            if begin_match:
                doc_key = get_doc_key(begin_match.group(1), begin_match.group(2))
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
            output_file.write("\n")
        else:
            assert get_doc_key(row[0], row[1]) == doc_key
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1


def output_conll_corefud(input_file, output_file, predictions, subtoken_map):
    prediction_map = {}
    for doc_key, clusters in predictions.items():
        start_map = collections.defaultdict(list)
        end_map = collections.defaultdict(list)
        word_map = collections.defaultdict(list)
        for cluster_id, mentions in enumerate(clusters):
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                if start == end:
                    word_map[start].append(cluster_id)
                else:
                    start_map[start].append((cluster_id, end))
                    end_map[end].append((cluster_id, start))
        for k,v in start_map.items():
            start_map[k] = [cluster_id for cluster_id, end in sorted(v, key=operator.itemgetter(1), reverse=True)]
        for k,v in end_map.items():
            end_map[k] = [cluster_id for cluster_id, start in sorted(v, key=operator.itemgetter(1), reverse=True)]
        prediction_map[doc_key] = (start_map, end_map, word_map)

    word_index = 0
    for line in input_file.readlines():
        row = line.split()
        if len(row) == 0:
            output_file.write("\n")
        elif row[0].startswith("#"):
            begin_match = re.match(BEGIN_DOCUMENT_REGEX_COREFUD, line)
            if begin_match:
                doc_key = begin_match.group(1)
                start_map, end_map, word_map = prediction_map[doc_key]
                word_index = 0
            output_file.write(line)
        else:
            coref_list = []
            if word_index in end_map:
                for cluster_id in end_map[word_index]:
                    coref_list.append("{})".format(cluster_id))
            if word_index in word_map:
                for cluster_id in word_map[word_index]:
                    coref_list.append("({})".format(cluster_id))
            if word_index in start_map:
                for cluster_id in start_map[word_index]:
                    coref_list.append("({}".format(cluster_id))

            if len(coref_list) == 0:
                row[-1] = "-"
            else:
                row[-1] = "|".join(coref_list)

            output_file.write("   ".join(row))
            output_file.write("\n")
            word_index += 1

def map_to_udapi(udapi_docs, predictions, subtoken_map):
    udapi_docs_map = {doc.meta["docname"]: doc for doc in udapi_docs}
    for doc_key, clusters in predictions.items():
        doc = udapi_docs_map[doc_key]
        udapi_words = [word for word in doc.nodes_and_empty]
        udapi_clusters = {}
        for cluster_id, mentions in enumerate(clusters):
            cluster = udapi_clusters.get(cluster_id)
            if cluster is None:
                cluster = CorefCluster(str(cluster_id))
                udapi_clusters[cluster_id] = cluster
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                mention = udapi.core.coref.CorefMention(words=udapi_words[start: end + 1], head=udapi_words[start], cluster=cluster)

        # doc._coref_clusters = {c._cluster_id: c for c in sorted(udapi_clusters.values())}
        doc._coref_clusters = udapi_clusters
        udapi.core.coref.store_coref_to_misc(doc)
        return udapi_docs


def official_conll_eval(gold_path, predicted_path, metric, official_stdout=True):
    cmd = ["conll-2012/scorer/v8.01/scorer.pl", metric, gold_path, predicted_path, "none"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    if stderr is not None:
        logger.error(stderr)

    if official_stdout:
        logger.info("Official result for {}".format(metric))
        logger.info(stdout)

    coref_results_match = re.match(COREF_RESULTS_REGEX, stdout)
    recall = float(coref_results_match.group(1))
    precision = float(coref_results_match.group(2))
    f1 = float(coref_results_match.group(3))
    return {"r": recall, "p": precision, "f": f1}


def evaluate_conll(gold_path, predictions, subtoken_maps, official_stdout=True, save_predictions=None):
    if save_predictions is None:
        prediction_file = tempfile.NamedTemporaryFile(delete=True, mode="w", encoding="utf-8")
    else:
        prediction_file = open(save_predictions, "w", encoding="utf-8")
    with open(gold_path, "r", encoding="utf-8") as gold_file:
        output_conll(gold_file, prediction_file, predictions, subtoken_maps)
    # logger.info("Predicted conll file: {}".format(prediction_file.name))
    results = {m: official_conll_eval(gold_file.name, prediction_file.name, m, official_stdout) for m in ("muc", "bcub", "ceafe") }
    prediction_file.close()
    return results

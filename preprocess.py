import argparse
import logging
import os
import sys
import re
import collections
import json
from transformers import BertTokenizer, AutoTokenizer
import conll
import util
import udapi_io
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def skip_doc(doc_key):
    return False


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class DocumentState(object):
    def __init__(self, key, path_length: int = 5):
        self.doc_key = key
        self.path_length = path_length

        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    if subtoken_info[4] == 'PRP':
                        self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = util.flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            if coref != '-':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        else:
                            cluster_id = int(part[1:])
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    else:
                        cluster_id = int(part[:-1])
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)

        # Sanity check
        # assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

    def finalize_from_udapi(self, udapi_doc):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        heads = {}
        final_heads = {}
        parents = []
        deprels = []
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            subdeprels = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                    subdeprels.append("<UNK>")
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append("SPEAKER1")
                    subdeprels.append(subtoken_info[-2])
                else:
                    speakers.append(speakers[-1])
                    subdeprels.append(subdeprels[-1])
                subtoken_idx += 1
            self.speakers += [speakers]
            deprels += [subdeprels]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = util.flatten(self.segment_info)
        word2subword = {}
        for word_index, word in enumerate(udapi_doc.nodes_and_empty):
            while subtokens_info[first_subtoken_idx] is None:
                first_subtoken_idx += 1
            subtoken_info = subtokens_info[first_subtoken_idx]
            corefs = word.coref_mentions
            word2subword[word_index] = first_subtoken_idx
            if word.ord != subtoken_info[0]:
                print("fd")
                pass
            if len(corefs) > 0:
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in corefs:
                    if "," in part.span:
                        continue    # Skip discontinuous mentions
                    cluster_id = part.entity.eid
                    if part.head == word:
                        heads[word.ord] = first_subtoken_idx
                    if part.span.split("-")[0] == str(word.ord):
                        if part.span.split("-")[-1] == str(word.ord):
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                            final_heads[str(first_subtoken_idx) + "-" + str(last_subtoken_idx)] = first_subtoken_idx
                        else:
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    elif part.span.split("-")[-1] == str(word.ord):
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
                        final_heads[str(start) + "-" + str(last_subtoken_idx)] = heads[part.head.ord]
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = util.flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = util.flatten(self.segment_subtoken_map)


        udapi_words = list(udapi_doc.nodes_and_empty)
        word_index = 0
        offset = 0
        parents = []
        for seg_info in self.segment_info:
            subparents = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    subparents.append(-1)
                elif subtoken_info is not None:  # First subtoken of each word
                    parent = udapi_words[word_index].parent
                    if parent is None or parent.form == "<ROOT>":
                        subparents.append(-1)
                    else:
                        subparents.append(word2subword[udapi_words.index(parent)])
                    word_index += 1
                else:
                    subparents.append(subparents[-1])

            ###### convert parents to paths
            subparents = np.array(subparents)

            # -1 to offset to enable indexing
            subparents[subparents == -1] = offset

            superparents = subparents
            tree_path = superparents
            for i in range(self.path_length-1):
                superparents = superparents[superparents-offset]
                tree_path = np.row_stack((tree_path, superparents))

            # mask invalid position with -1
            tree_path[tree_path == offset] = -1

            parents += [tree_path]
            offset = len(seg_info)
        # Sanity check
        # assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(util.flatten(self.segments))
        assert num_all_seg_tokens == len(util.flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns,
            "parents": parents,
            "deprels": deprels,
            "heads": final_heads
        }


def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.error('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def get_document(doc_key, language, seg_len, tokenizer, udapi_document=None):
    """ Process raw input to finalized documents """
    document_state = DocumentState(doc_key)
    word_idx = -1

    # Build up documents
    last_ord = 0
    for node in udapi_document.nodes_and_empty:
        if last_ord >= node.ord:
            document_state.sentence_end[-1] = True
            # assert len(row) >= 12
        word_idx += 1
        word = normalize_word(node.form, language)
        subtokens = tokenizer.tokenize(word)
        document_state.tokens.append(word)
        document_state.token_end += [False] * (len(subtokens) - 1) + [True]
        for idx, subtoken in enumerate(subtokens):
            document_state.subtokens.append(subtoken)
            info = None if idx != 0 else ([node.ord] + [node.form] + [node.parent] + [node.udeprel] + [len(subtokens)])
            document_state.info.append(info)
            document_state.sentence_end.append(False)
            document_state.subtoken_map.append(word_idx)
        last_ord = node.ord
    document_state.sentence_end[-1] = True

    # Split documents
    constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
    if udapi_document is not None:
        document = document_state.finalize_from_udapi(udapi_document)
    else:
        document = document_state.finalize()
    return document


def minimize_partition(partition, extension, args, tokenizer):
    input_path = os.path.join(args.data_dir, f'{args.language}-{partition}.{extension}')
    output_path = os.path.join(args.data_dir, f'{args.language}-{partition}.{args.max_segment_len}.jsonlines')
    doc_count = 0
    logger.info(f'Minimizing {input_path}...')

    # Write documents
    with open(output_path, 'w') as output_file:
        if "joined_languages" in args:
            udapi_documents = []
            for i in range(len(args.joined_languages)):
                input_path = os.path.join(args.joined_dirs[i], f'{args.joined_languages[i]}-{partition}.{extension}')
                udapi_documents.extend(udapi_io.read_data(input_path))
        else:
            udapi_documents = udapi_io.read_data(input_path)
        for doc in udapi_documents:
            document = get_document(doc.meta["docname"], args.language, args.max_segment_len, tokenizer, udapi_documents[doc_count])
            output_file.write(json.dumps(document))
            output_file.write('\n')
            doc_count += 1
    logger.info(f'Processed {doc_count} documents to {output_path}')


def minimize_language(args):
    tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_name)

    # minimize_partition('dev', 'v4_gold_conll', args, tokenizer)
    # minimize_partition('test', 'v4_gold_conll', args, tokenizer)
    # minimize_partition('train', 'v4_gold_conll', args, tokenizer)


    minimize_partition('test', 'conllu', args, tokenizer)
    minimize_partition('dev', 'conllu', args, tokenizer)
    minimize_partition('train', 'conllu', args, tokenizer)


if __name__ == '__main__':
    config_name = sys.argv[1]
    config = util.initialize_config(config_name)

    os.makedirs(config.data_dir, exist_ok=True)

    minimize_language(config)

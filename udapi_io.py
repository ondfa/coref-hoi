import logging
import os
import shutil

import udapi
import udapi.core
from udapi.block.corefud.movehead import MoveHead
from udapi.block.corefud.guessspan import GuessSpan
from udapi.block.corefud.fixinterleaved import FixInterleaved

from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter
from udapi.core.coref import CorefEntity

logger = logging.getLogger(__name__)

def read_data(file):
    move_head = MoveHead()
    docs = ConlluReader(files=file, split_docs=True).read_documents()
    level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    for doc in docs:
        move_head.run(doc)
    logging.getLogger().setLevel(level)
    return docs

def write_data(docs, f):
    level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    writer = ConlluWriter(filehandle=f)
    for doc in docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
    # writer.after_process_document(None)
    logging.getLogger().setLevel(level)


def map_to_udapi(udapi_docs, predictions, subtoken_map, doc_span_to_head=None, use_guess_span=False):
    level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    entities = 1
    udapi_docs_map = {doc.meta["docname"]: doc for doc in udapi_docs}
    docs = []
    move_head = MoveHead()
    guess_span = GuessSpan()
    fix_interleaved = FixInterleaved()
    for doc_key, clusters in predictions.items():
        if doc_key not in udapi_docs_map:
            continue
        doc = udapi_docs_map[doc_key]
        udapi_words = [word for word in doc.nodes_and_empty]
        for word in udapi_words:
            word.misc = {}
        doc._eid_to_entity = {}
        for mentions in clusters:
            entity = doc.create_coref_entity(eid="e" + str(entities))
            entities += 1
            for i, (start, end) in enumerate(mentions):
                if doc_span_to_head and doc_key in doc_span_to_head:
                    heads = doc_span_to_head[doc_key][str(start) + "-" + str(end)]
                    heads = [subtoken_map[doc_key][start + head] for head in heads]
                    head_words = [udapi_words[head] for head in heads]
                    entity.create_mention(words=head_words)
                else:
                    start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                    entity.create_mention(words=udapi_words[start: end + 1])
        move_head.run(doc)
        if use_guess_span:
            guess_span.run(doc)
            fix_interleaved.run(doc)
        udapi.core.coref.store_coref_to_misc(doc)
        docs.append(doc)
    logging.getLogger().setLevel(level)
    return docs


def filter_long_mentions(udapi_docs, max_len=1):
    for doc in udapi_docs:
        for entity in doc.coref_entities:
            entity.mentions = [mention for mention in entity.mentions if len(mention.words) <= max_len]


def convert_all_to_text(config):
    input_path = config["conll_eval_path"]
    gold_path = os.path.join(os.path.split(input_path)[0], "gold", os.path.split(input_path)[-1])
    if os.path.exists(gold_path):
        docs = read_data(gold_path)
        out_file = gold_path.replace(".conllu", "-GOLD.txt")
        convert_to_text(docs, out_file)

    docs = read_data(input_path)
    out_file = input_path.replace("conllu", "txt")
    convert_to_text(docs, out_file, solve_empty_nodes=False)

    out_file = input_path.replace(".conllu", "-BLIND.txt")
    convert_to_text(docs, out_file, mark_entities=False, solve_empty_nodes=False)
    
    out_file = input_path.replace(".conllu", "-ZEROS-BLIND.txt")
    convert_to_text(docs, out_file, mark_entities=False)
    
    out_file = input_path.replace(".conllu", "-ZEROS.txt")
    convert_to_text(docs, out_file)

    input_path = config["conll_test_path"]
    docs = read_data(input_path)
    out_file = config["conll_test_path"].replace(".conllu", "-BLIND.txt")
    convert_to_text(docs, out_file, solve_empty_nodes=False)
    
    out_file = config["conll_test_path"].replace(".conllu", "-ZEROS-BLIND.txt")
    convert_to_text(docs, out_file)
    # input_path = config["conll_eval_path"].replace("-dev", "-train")
    # docs = read_data(input_path)
    # out_file = input_path.replace("conllu", "txt")
    # convert_to_text(docs, out_file)

    # TRAIN
    input_path = os.path.join(config["data_dir"], '..', f'{config["language"]}-train.conllu')
    docs = read_data(input_path)
    out_file = input_path.replace("conllu", "txt")
    convert_to_text(docs, out_file, solve_empty_nodes=False)

    out_file = input_path.replace(".conllu", "-ZEROS.txt")
    convert_to_text(docs, out_file)

    out_file = input_path.replace(".conllu", "-BLIND.txt")
    convert_to_text(docs, out_file, mark_entities=False, solve_empty_nodes=False)

    out_file = input_path.replace(".conllu", "-ZEROS-BLIND.txt")
    convert_to_text(docs, out_file, mark_entities=False)


def convert_to_text(docs, out_file, solve_empty_nodes=True, mark_entities=True, sequential_ids=False):
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            eids = {}
            out_words = []
            if solve_empty_nodes:
                udapi_words = [word for word in doc.nodes_and_empty]
            else:
                udapi_words = [word for word in doc.nodes]
            for word in udapi_words:
                out_word = word.form
                if word.lemma.startswith("#") and solve_empty_nodes:
                    out_word += word.lemma
                mentions = []
                if mark_entities:
                    for mention in set(word.coref_mentions):
                        if sequential_ids:
                            if mention.entity.eid not in eids:
                                eids[mention.entity.eid] = f"e{len(eids) + 1}"
                            eid = eids[mention.entity.eid]
                        else:
                            eid = mention.entity.eid
                        span = mention.span
                        if "," in span:
                            span = span.split(",")[0]
                        mention_start = float(span.split("-")[0])
                        mention_end = float(span.split("-")[1]) if "-" in span else mention_start
                        if mention_start == float(word.ord) and mention_end == float(word.ord):
                            mentions.append(f"({eid})")
                        elif mention_start == float(word.ord):
                            mentions.append(f"({eid}")
                        elif mention_end == float(word.ord):
                            mentions.append(f"{eid})")
                if len(mentions) > 0:
                    out_words.append(f"{out_word}|{','.join(mentions)}")
                else:
                    out_words.append(out_word)
            f.write(" ".join(out_words) + "\n")
    output_dir = "data/UD/text/"
    os.makedirs(output_dir, exist_ok=True)
    copy_dest = os.path.join(output_dir, out_file.split("/")[-1])
    shutil.copyfile(out_file, copy_dest)


def convert_text_to_conllu(text_docs, conllu_skeleton_file, out_file, solve_empty_nodes=True):
    udapi_docs = read_data(conllu_skeleton_file)
    for doc in udapi_docs:
        doc._eid_to_entity = {}
    assert len(udapi_docs) == len(text_docs)
    for text, udapi_doc in zip(text_docs, udapi_docs):
        if solve_empty_nodes:
            udapi_words = [word for word in udapi_doc.nodes_and_empty]
        else:
            udapi_words = [word for word in udapi_doc.nodes]
        words = text.split(" ")
        assert len(udapi_words) == len(words)
        mention_starts = {}
        entities = {}
        for word, udapi_word in zip(words, udapi_words):
            if "|" in word:
                mentions = word.split("|")[1].split(",")
                for mention in mentions:
                    eid = mention.replace("(", "").replace(")", "")
                    if eid not in entities:
                        entities[eid] = udapi_doc.create_coref_entity(eid=eid)
                    if mention.startswith("("):
                        if eid in mention_starts:
                            logger.warning(f"WARNING: Multiple mentions of the same entity opened. EID: {eid}")
                        mention_starts[eid] = udapi_word.ord - 1
                    if mention[-1] == ")":
                        if eid not in mention_starts:
                            logger.warning(f"WARNING: Closing mention which was not opened. EID: {eid}")
                            continue
                        entities[eid].create_mention(words=udapi_words[mention_starts[eid]: udapi_word.ord])
                        del mention_starts[eid]
    with open(out_file, "w", encoding="utf-8") as f:
        write_data(udapi_docs, f)


def highlight_differences(gold_file, first_file, second_file, skip_singletons=False):
    gold_data = read_data(gold_file)
    first_data = read_data(first_file)
    second_data = read_data(second_file)
    for gold_doc, first_doc, second_doc in zip(gold_data, first_data, second_data):
        gold_mentions = list(gold_doc.coref_mentions)
        first_mentions = list(first_doc.coref_mentions)
        second_mentions = list(second_doc.coref_mentions)
        j = k = 0
        num_gold = 0
        first_correct_mentions = second_correct_mentions = 0
        for i in range(len(gold_mentions)):
            if skip_singletons and len(gold_mentions[i].entity.mentions) == 1:
                continue
            num_gold += 1
            prev_j, prev_k = j, k
            prev_correct_j, prev_correct_k = first_correct_mentions, second_correct_mentions
            if j < len(first_mentions):
                if gold_mentions[i].head.form != first_mentions[j].head.form:
                    while j < len(first_mentions) and first_mentions[j].head.precedes(gold_mentions[i].head):
                        j += 1
                if j < len(first_mentions) and gold_mentions[i].head.form == first_mentions[j].head.form:
                    j += 1
                    first_correct_mentions += 1
            if k < len(second_mentions):
                if gold_mentions[i].head.form != second_mentions[k].head.form:
                    while k < len(second_mentions) and second_mentions[k].head.precedes(gold_mentions[i].head):
                        k += 1
                if k < len(second_mentions) and gold_mentions[i].head.form == second_mentions[k].head.form:
                    k += 1
                    second_correct_mentions += 1
            # if (first_correct_mentions == prev_correct_j or second_correct_mentions == prev_correct_k) and (j < len(first_mentions) or k < len(second_mentions)):
            #     print("Difference: GOLD, first, second")
            #     print(str(gold_mentions[i].head))
            #     if j != prev_j and j < len(first_mentions):
            #         print(f"first: {first_mentions[j].head}")
            #     if k != prev_k and k < len(second_mentions):
            #         print(f"second {second_mentions[k].head}")
        print(f"Gold mentions: {num_gold}, first correct: {first_correct_mentions}, second correct: {second_correct_mentions}")
        print(f"First added {len(first_mentions) - first_correct_mentions} mentions. Second added {len(second_mentions) - second_correct_mentions} mentions.")





if __name__ == '__main__':
    highlight_differences("data/UD/CorefUD-1.1-test-final/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu", "data/cases/base/cs_pdt-corefud-dev.conllu", "data/cases/best/cs_pdt-corefud-dev.conllu", skip_singletons=True)
    exit(0)
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    with open("dev.conllu", "wt", encoding="utf-8") as f:
        write_data(docs, f)

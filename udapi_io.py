import logging
import os
import shutil

import udapi
import udapi.core
from udapi.block.corefud.movehead import MoveHead
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter
from udapi.core.coref import CorefEntity


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


def map_to_udapi(udapi_docs, predictions, subtoken_map, doc_span_to_head=None):
    level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)
    entities = 1
    udapi_docs_map = {doc.meta["docname"]: doc for doc in udapi_docs}
    docs = []
    move_head = MoveHead()
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
        udapi.core.coref.store_coref_to_misc(doc)
        docs.append(doc)
    logging.getLogger().setLevel(level)
    return docs


def filter_long_mentions(udapi_docs, max_len=1):
    for doc in udapi_docs:
        for entity in doc.coref_entities:
            entity.mentions = [mention for mention in entity.mentions if len(mention.words) <= max_len]


def convert_all_to_text(config):
    # docs = read_data(config["conll_eval_path"])
    # out_file = config["conll_eval_path"].replace("conllu", "txt")
    # convert_to_text(docs, out_file)

    input_path = config["conll_eval_path"]
    gold_path = os.path.join(os.path.split(input_path)[0], "gold", os.path.split(input_path)[-1])
    if os.path.exists(gold_path):
        docs = read_data(gold_path)
        out_file = gold_path.replace(".conllu", "-GOLD.txt")
        convert_to_text(docs, out_file)

    docs = read_data(config["conll_test_path"])
    out_file = config["conll_eval_path"].replace("conllu", "txt")
    convert_to_text(docs, out_file)

    # input_path = config["conll_eval_path"].replace("-dev", "-train")
    # docs = read_data(input_path)
    # out_file = input_path.replace("conllu", "txt")
    # convert_to_text(docs, out_file)

def convert_to_text(docs, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in docs:
            out_words = []
            udapi_words = [word for word in doc.nodes_and_empty]
            for word in udapi_words:
                out_word = word.form
                mentions = []
                for mention in set(word.coref_mentions):
                    span = mention.span
                    if "," in span:
                        span = span.split(",")[0]
                    mention_start = float(span.split("-")[0])
                    mention_end = float(span.split("-")[1]) if "-" in span else mention_start
                    if mention_start == float(word.ord) and mention_end == float(word.ord):
                        mentions.append(f"({mention.entity.eid})")
                    elif mention_start == float(word.ord):
                        mentions.append(f"({mention.entity.eid}")
                    elif mention_end == float(word.ord):
                        mentions.append(f"{mention.entity.eid})")
                if len(mentions) > 0:
                    out_words.append(f"{out_word}|{','.join(mentions)}")
                else:
                    out_words.append(out_word)
            f.write(" ".join(out_words) + "\n")
    output_dir = "data/UD/text/"
    os.makedirs(output_dir, exist_ok=True)
    copy_dest = os.path.join(output_dir, out_file.split("/")[-1])
    shutil.copyfile(out_file, copy_dest)


if __name__ == '__main__':
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    with open("dev.conllu", "wt", encoding="utf-8") as f:
        write_data(docs, f)

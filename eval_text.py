import logging

import udapi
from udapi.block.corefud.movehead import MoveHead
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter

import subprocess

from udapi.core.coref import BridgingLinks

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

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


def convert_text_to_conllu(text_docs, conllu_skeleton_file, out_file, solve_empty_nodes=True):
    udapi_docs = read_data(conllu_skeleton_file)
    # udapi_docs2 = read_data(conllu_skeleton_file)
    move_head = MoveHead()
    for doc in udapi_docs:
        doc._eid_to_entity = {}
    assert len(udapi_docs) == len(text_docs)
    for text, udapi_doc in zip(text_docs, udapi_docs):
        if solve_empty_nodes:
            udapi_words = [word for word in udapi_doc.nodes_and_empty]
        else:
            udapi_words = [word for word in udapi_doc.nodes]
        for word in udapi_doc.nodes_and_empty:
            word.misc = {}
        words = text.split(" ")
        if len(udapi_words) != len(words):
            continue
        assert len(udapi_words) == len(words)
        mention_starts = {}
        entities = {}
        for i, (word, udapi_word) in enumerate(zip(words, udapi_words)):
            if word.split("|")[0] != udapi_word.form:
                logger.warning(f"WARNING: words do not match. DOC: {udapi_doc.meta['docname']}, word1: {word.split('|')[0]}, word2: {udapi_word.form}")
            if "|" in word:
                mentions = word.split("|")[1].replace("-", ",").split(",")
                for mention in mentions:
                    eid = mention.replace("(", "").replace(")", "")
                    if len(eid) == 0:
                        continue
                    if eid not in entities:
                        entities[eid] = udapi_doc.create_coref_entity(eid=eid)
                    if mention.startswith("("):
                        if eid in mention_starts:
                            logger.warning(f"WARNING: Multiple mentions of the same entity opened. DOC: {udapi_doc.meta['docname']}, EID: {eid}")
                            # continue
                        mention_starts[eid] = i
                    if mention[-1] == ")":
                        if eid not in mention_starts:
                            logger.warning(f"WARNING: Closing mention which was not opened. DOC: {udapi_doc.meta['docname']}, EID: {eid}")
                            continue
                        entities[eid].create_mention(words=udapi_words[mention_starts[eid]: i + 1])
                        del mention_starts[eid]
        udapi.core.coref.store_coref_to_misc(udapi_doc)
        move_head.run(udapi_doc)
    # debug_udapi(udapi_docs, udapi_docs2)
    with open(out_file, "w", encoding="utf-8") as f:
        write_data(udapi_docs, f)

def remove_empty(in_file, out_file):
    udapi_docs = read_data(in_file)
    for doc in udapi_docs:
        for entity in doc.coref_entities:
            entity.split_ante = []
        for mention in doc.coref_mentions:
            mention.bridging._data = []
        for word in doc.nodes_and_empty:
            word.misc = {}
            if word.ord - int(word.ord) > 0:
                for entity in word.coref_entities:
                    if entity.eid in doc._eid_to_entity:
                        del doc._eid_to_entity[entity.eid]
        udapi.core.coref.store_coref_to_misc(doc)
    with open(out_file, "w", encoding="utf-8") as f:
        write_data(udapi_docs, f)

def evaluate_coreud(gold_path, pred_path, python_cmd):
    cmd = [python_cmd, "corefud-scorer/corefud-scorer.py", gold_path, pred_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    # if stderr is not None:
    #     logger.error(stderr)
    logger.info("Official result for {}".format(pred_path))
    logger.info(stdout)
    import re
    result = re.search(r"CoNLL score: (\d+\.?\d*)", stdout)
    if result is None:
        score = 0.0
    else:
        score = float(result.group(1))

    cmd = ["python", "corefud-scorer/corefud-scorer.py", gold_path, pred_path, "-s"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    stdout = stdout.decode("utf-8")
    # if stderr is not None:
    #     logger.error(stderr)
    logger.info("Official result with singletons for {}".format(pred_path))
    logger.info(stdout)
    result = re.search(r"CoNLL score: (\d+\.?\d*)", stdout)
    if result is None:
        score_with_singletons = 0.0
    else:
        score_with_singletons = float(result.group(1))
    return score, score_with_singletons


def eval_text(text_docs, gold_file, solve_empty_nodes=True):
    pred_file = f"{gold_file.replace('.conllu', '')}-pred.conllu"
    convert_text_to_conllu(text_docs, gold_file, pred_file, solve_empty_nodes)
    # if not solve_empty_nodes:
    #     eval_tmp = gold_file.replace(".conllu", "-nonempty.conllu")
    #     remove_empty(gold_file, eval_tmp)
    #     gold_file = eval_tmp
    evaluate_coreud(gold_file, pred_file, "python3.10")
    # evaluate_coreud(gold_file, gold_file, "python3.10")


def debug_udapi(udapi_docs1, udapi_docs2):
    for doc1, doc2 in zip(udapi_docs1, udapi_docs2):
        for e1, e2 in zip(doc1.coref_entities, doc2.coref_entities):
            for m1, m2 in zip(e1.mentions, e2.mentions):
                if m1.span != m2.span:
                    logger.error("spans do not match")


if __name__ == '__main__':
    with open("text_eval/gpt/cs_pcedt-corefud-test-gpt4o-mini-cleaned.txt", encoding="utf-8") as f:
        text_docs = f.read().splitlines()
        eval_text(text_docs, "text_eval/gpt/cs_pcedt-corefud-test.conllu", solve_empty_nodes=False)

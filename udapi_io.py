import udapi
import udapi.core
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter
from udapi.core.coref import CorefEntity


def read_data(file):
    return ConlluReader(files=file, split_docs=True).read_documents()

def write_data(docs, f):
    writer = ConlluWriter(filehandle=f)
    for doc in docs:
        writer.before_process_document(doc)
        writer.process_document(doc)
    # writer.after_process_document(None)


def map_to_udapi(udapi_docs, predictions, subtoken_map, doc_span_to_head=None):
    entities = 1
    udapi_docs_map = {doc.meta["docname"]: doc for doc in udapi_docs}
    docs = []
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
        udapi.core.coref.store_coref_to_misc(doc)
        docs.append(doc)
    return docs

if __name__ == '__main__':
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    with open("dev.conllu", "wt", encoding="utf-8") as f:
        write_data(docs, f)

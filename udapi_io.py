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


def map_to_udapi(udapi_docs, predictions, subtoken_map):
    udapi_docs_map = {doc.meta["docname"]: doc for doc in udapi_docs}
    for doc_key, clusters in predictions.items():
        doc = udapi_docs_map[doc_key]
        udapi_words = [word for word in doc.nodes_and_empty]
        doc._eid_to_entity = {}
        for mentions in clusters:
            entity = doc.create_coref_entity()
            for start, end in mentions:
                start, end = subtoken_map[doc_key][start], subtoken_map[doc_key][end]
                entity.create_mention(words=udapi_words[start: end + 1])
        udapi.core.coref.store_coref_to_misc(doc)
    return udapi_docs

if __name__ == '__main__':
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    with open("dev.conllu", "wt", encoding="utf-8") as f:
        write_data(docs, f)

import udapi
import udapi.core
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter
from udapi.core.coref import CorefCluster


def read_data(file):
    reader = ConlluReader(files=file, split_docs=True)
    docs = []
    while not reader.finished:
        doc = udapi.Document(None)
        reader.process_document(doc)
        docs.append(doc)
    # for doc in docs:
    #     for word in doc.nodes:
    #         pass
    return docs

def write_data(docs, file):
    with open(file, "wt", encoding="utf-8") as f:
        writer = ConlluWriter(filehandle=f)
        for doc in docs:
            writer.before_process_document(doc)
            writer.process_document(doc)
        writer.after_process_document(None)


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

if __name__ == '__main__':
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    write_data(docs, "dev.conllu")

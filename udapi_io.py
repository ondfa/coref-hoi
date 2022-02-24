import udapi
from udapi.block.read.conllu import Conllu as ConlluReader
from udapi.block.write.conllu import Conllu as ConlluWriter


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

if __name__ == '__main__':
    docs = read_data("data/UD/CorefUD-0.1-public/data/CorefUD_Czech-PDT/cs_pdt-corefud-dev.conllu")
    write_data(docs, "dev.conllu")

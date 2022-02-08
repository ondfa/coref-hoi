import csv


def readUD(file):
    with open(file) as f:
        # reader = csv.reader(f, delimiter="\t")
        docs = []
        paragraphs = []
        sentences = []
        sentence = []
        doc_id = ""
        sent_id = ""
        par_id = ""
        for line in f.read().splitlines():
            if not line:
                continue
            line = line.split("\t")
            if line[0].startswith("# newdoc"):
                doc_id = line[0][line[0].index("id =") + 5:]
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []

                if len(sentences) > 0:
                    paragraphs.append(sentences)
                sentences = []

                if len(paragraphs) > 0:
                    docs.append(paragraphs)
                paragraphs = []

            if line[0].startswith("# newpar"):
                par_id = line[0][line[0].index("id =") + 5:]
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []

                if len(sentences) > 0:
                    paragraphs.append(sentences)
                sentences = []

            if line[0].startswith("# sent_id"):
                sent_id = line[0][line[0].index("id =") + 5:]
                if len(sentence) > 0:
                    sentences.append(sentence)
                sentence = []

            if line[0].startswith("#"):
                continue
            word = {}
            word["index"] = line[0]
            word["form"] = line[1]
            word["lemma"] = line[2]
            word["pos"] = line[3]
            word["doc_id"] = doc_id
            # word["par_id"] = par_id[par_id.index("-p") + 2:]
            word["par_id"] = "000"
            word["sent_id"] = sent_id
            word["cluster"] = line[9]
            sentence.append(word)

    return docs


def parse_properties(field):
    if field == "_":
        return {}
    properties = [prop.split("=") for prop in field.split("|")]
    return {kv[0]: kv[1] for kv in properties}

def find_index(index, sentence, start):
    for i in range(start, len(sentence)):
        if sentence[i]["index"] == index:
            return i
    if start > 0:
        return find_index(index, sentence, 0)
    return -1

def add_cluster(cluster_string, word):
    if "clusterOnto" not in word or word["clusterOnto"] == "-":
        word["clusterOnto"] = cluster_string
    else:
        word["clusterOnto"] += "|" + cluster_string

def parse_sentence(sentence):
    for i in range(len(sentence)):
        sentence[i]["speaker"] = "SPEAKER1"
        if "clusterOnto" not in sentence[i]:
            sentence[i]["clusterOnto"] = "-"
        if sentence[i]["cluster"] and sentence[i]["cluster"] != "_":
            props = parse_properties(sentence[i]["cluster"])
            if "MentionSpan" not in props:
                continue
            for span in props["MentionSpan"].split(","):
                mention = span.split("-")
                index = find_index(mention[0], sentence, i)
                add_cluster("(" + props["ClusterId"][1:], sentence[index])
                if len(mention) > 1:
                    index = find_index(mention[1], sentence, i)
                    add_cluster(props["ClusterId"][1:], sentence[index])
                sentence[index]["clusterOnto"] += ")"
    return sentence

def parse(docs):
    for doc in docs:
        for paragraph in doc:
            for sentence in paragraph:
                parse_sentence(sentence)
    return docs

def write(docs, file):
    with open(file, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for doc in docs:
            writer.writerow([f"#begin document ({doc[0][0][0]['doc_id']}); part 000"])
            for paragraph in doc:
                for sentence in paragraph:
                    for word in sentence:
                        writer.writerow([word["doc_id"], word["par_id"], word["index"], word["form"], word["pos"], "-", "-", "-", "-", word["speaker"], "*", word["clusterOnto"]])
                    writer.writerow([])
            writer.writerow([f"#end document"])
    return


def convert(input, output):
    examples = readUD(input)
    examplesOnto = parse(examples)
    write(examplesOnto, output)


if __name__ == '__main__':
    examples = readUD("./data/UD/CorefUD-0.1-public/data/CorefUD_German-ParCorFull/de_parcorfull-corefud-train.conllu")
    examplesOnto = parse(examples)
    write(examplesOnto, "data/UD/train.german-parcor.conllu")

    examples = readUD("./data/UD/CorefUD-0.1-public/data/CorefUD_German-ParCorFull/de_parcorfull-corefud-dev.conllu")
    examplesOnto = parse(examples)
    write(examplesOnto, "data/UD/dev.german-parcor.conllu")
    write(examplesOnto, "data/UD/test.german-parcor.conllu")


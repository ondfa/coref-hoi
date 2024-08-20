

def split_document(input_file, output_file, docname, split_points):
    in_doc = False
    split_index = 0
    par_index = 1
    sent_index = 1
    act_doc_name = ""
    act_par_name = ""
    global_entity_line = ""
    with open(input_file, encoding="utf-8") as r:
        with open(output_file, "w", encoding="utf-8") as w:
            for line in r:
                if line.startswith("# global.Entity "):
                    global_entity_line = line
                if line.startswith(f"# newdoc id = {docname}"):
                    in_doc = True
                elif line.startswith("# newdoc"):
                    in_doc = False
                if in_doc and split_index < len(split_points) and line.startswith(split_points[split_index]):
                    split_index += 1
                    par_index = sent_index = 1
                    act_doc_name = f"{docname}-{split_index + 1}"
                    w.write(f"# newdoc id = {act_doc_name}\n")
                    w.write(global_entity_line)
                if not in_doc or split_index == 0:
                    w.write(line)
                    continue
                if line.startswith("# newpar"):
                    sent_index = 1
                    act_par_name = f"{act_doc_name}-p{par_index}"
                    par_index += 1
                    w.write(f"# newpar id = {act_par_name}\n")
                elif line.startswith("# sent_id"):
                    w.write(f"# sent_id = {act_par_name}-s{sent_index}\n")
                    sent_index += 1
                else:
                    w.write(line)


def main():
    split_document("data/UD/CorefUD-1.1-test-final/data/CorefUD_French-Democrat/fr_democrat-corefud-dev.conllu", "data/UD/CorefUD-1.1-test-final/data/CorefUD_French-Democrat/fr_democrat-corefud-dev-fixed.conllu", "mademoisellefifi-1", ["# newpar id = mademoisellefifi-1-p98", "# newpar id = mademoisellefifi-1-p146"])


if __name__ == '__main__':
    main()

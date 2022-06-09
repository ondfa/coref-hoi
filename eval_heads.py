import sys
import os
import util
import udapi_io
import sentencepiece as spm


def eval_heads(config):
    path = os.path.join(config["data_dir"], f'{config["language"]}-dev.conllu')
    docs = udapi_io.read_data(path)
    all_heads = []
    count = 0
    dupl_count = 0
    for doc in docs:
        entity_heads = {}
        all_heads.append(entity_heads)
        for eid, entity in doc.eid_to_entity.items():
            for mention in entity.mentions:
                count += 1
                if mention.head not in entity_heads:
                    entity_heads[mention.head] = set()
                else:
                    # print("multiple mentions with the same head: " + str(eid))
                    dupl_count += 1
                entity_heads[mention.head].add(mention)
    return all_heads, dupl_count / count



if __name__ == '__main__':
    config_name = sys.argv[1]
    config = util.initialize_config(config_name)
    os.makedirs(config.data_dir, exist_ok=True)
    heads, count = eval_heads(config)
    # print(count)
    count = 0
    multiple = 0
    all = 0

    for doc in heads:
        entities = set()
        count += sum([len(l) for l in doc.values()])
        multiple += sum([len(l) for l in doc.values() if len(l) > 1])
        for l in doc.values():
            if len(l) > 1:
                entities.update([mention.entity for mention in l])
        all += sum([len(entity.mentions) for entity in entities])
    print(multiple / count)
    print(all / count)

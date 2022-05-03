import ujson


def load_encoding(input_path: str, name: str = "encodings.json") -> dict:
    personkey2name = {}
    with open(input_path + "german_persons.json", 'r', encoding='utf-8') as f:
        for line in f:
            line = ujson.loads(line)
            personkey2name[line['key']] = line['author']
    with open(input_path + name, 'r', encoding='utf-8') as f:
        encodings = ujson.load(f)
    encodings2name = {}
    for k, v in encodings.items():
        if k in personkey2name:
            encodings2name[v] = personkey2name[k]
    return encodings2name


def load_embedding(input_path: str, type: str) -> dict:
    if type == "hope":
        with open(input_path, 'r') as f:
            return ujson.load(f)
    else:  # will be deepwalk or node2vec
        with open(input_path, 'r') as f:
            lines = f.readlines()
        lines = lines[1:]  # delete first row
        lines = [line.strip().split(" ") for line in lines]
        return {int(line[0]): [float(element) for element in line[1:]] for line in lines}


def parse_id2names(embedding: dict, encoding2name: dict) -> dict:
    name2embedding = {}
    for k, v in embedding.items():
        k = int(k)
        if k in encoding2name:
            names = encoding2name[k] if isinstance(encoding2name[k], list) else [encoding2name[k]]
            for name in names:
                name2embedding[name] = v
    return name2embedding


def save_embedding(name2embedding: dict, output_path: str):
    output_path = output_path.replace(".embedding", "").replace(".json", "")
    with open(output_path + ".csv", 'w', encoding='utf-8') as f:
        for k, v in name2embedding.items():
            f.write(k + ";" + ",".join([str(x) for x in v]) + "\n")


graph_embeddings = {"deepwalk": ["deepwalk_all.embedding", "deepwalk_cc.embedding"],
                    "hope": ["hope_all.json", "hope_cc.json"],
                    "node2vec": ["german_ai_cc_p_large+q_small.embedding", "german_ai_cc_p_small+q_large.embedding",
                                 "german_ai_p_large+q_small.embedding", "german_ai_p_small+q_large.embedding"]}

graph_embeddings_split = {"deepwalk": [("split_2_train.embedding", "deepwalk_2.embedding"),
                                       ("split_5_train.embedding", "deepwalk_5.embedding")],
                          "hope": [("split_2_train.json", "hope_2.embedding"),
                                   ("split_2_train.json", "hope_5.embedding")],
                          "node2vec": [("split_2_train_p_large+q_small.embedding", "node2vec_largeP_2.embedding"),
                                       ("split_2_train_p_small+q_large.embedding", "node2vec_smallP_2.embedding"),
                                       ("split_5_train_p_large+q_small.embedding", "largeP_5.embedding"),
                                       ("split_5_train_p_small+q_large.embedding", "node2vec_smallP_5.embedding")]}
path = "../Max/data/"

if __name__ == '__main__':
    encoding2name_2 = load_encoding(input_path=path, name="split_2_econding.json")
    encoding2name_5 = load_encoding(input_path=path, name="split_5_econding.json")
    for embedding_type in graph_embeddings_split.keys():
        print("Loading ", embedding_type + " embeddings. ")
        for embedding_name in graph_embeddings_split[embedding_type]:
            encoding2embedding = load_embedding(input_path=path + embedding_type + "/" + embedding_name[0], type=embedding_type)
            if "2" in embedding_name[0]:
                name2embedding = parse_id2names(embedding=encoding2embedding, encoding2name=encoding2name_2)
            else:
                name2embedding = parse_id2names(embedding=encoding2embedding, encoding2name=encoding2name_5)
            save_embedding(name2embedding=name2embedding, output_path="data/" + embedding_name[1])
    print("Finished. ")

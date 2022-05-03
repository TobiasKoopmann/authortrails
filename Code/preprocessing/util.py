import ujson
import numpy as np

from tqdm import tqdm


def from_paper2author(input_meta: str = "./data/large_data/german_ai_community_dataset.json",
                      input_paper: str = "./data/large_data/scibert_paper_embeddings.csv",
                      output_path: str = "./data/large_data/scibert_author_embeddings.csv",
                      years_remove: int = None, filter_authors: bool = False):
    with open(input_meta, 'r', encoding='utf-8') as f:
        meta_data = f.readlines()
    print("Read input meta data. ")
    meta_data = [ujson.loads(x) for x in meta_data]
    if years_remove:
        meta_data = [x for x in meta_data if int(x['year']) <= years_remove]
    meta_data = [a for b in [[(author, paper['key']) for author in paper['author']] for paper in meta_data] for a in b]
    author_keys = {}
    for data in tqdm(meta_data):
        if data[0] in author_keys:
            author_keys[data[0]].extend([data[1]])
        else:
            author_keys[data[0]] = [data[1]]
    if filter_authors:
        with open("./data/german_persons.json", "r", encoding="utf-8") as f:
            tmp = f.readlines()
        tmp = [ujson.loads(x)['author'] for x in tmp]
        tmp = set([a for b in [x if isinstance(x, list) else [x] for x in tmp] for a in b])
        removes = []
        for author_key in author_keys.keys():
            if author_key not in tmp:
                removes.append(author_key)
        for rem in removes:
            del author_keys[rem]
    print("Having ", len(author_keys), "authors. ")
    paper_embeddings = {}
    with open(input_paper, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(";")
            paper_embeddings[line[0]] = np.array([float(x) for x in line[1].split(",")])
    with open(output_path, 'w', encoding='utf-8') as f:
        for author, keys in tqdm(author_keys.items()):
            vecs = np.mean([paper_embeddings[key] for key in keys if key in paper_embeddings], axis=0)
            if len(vecs.shape) != 0:
                f.write(author + ";" + ",".join([str(x) for x in list(vecs)]) + "\n")


def extract_german_ai_authors(input_embeddings: str, german_ai_authors: str, output_path: str):
    german_ai_author_names = set()
    with open(german_ai_authors, 'r', encoding='utf-8') as f:
        for line in f:
            author = ujson.loads(line)['author']
            if isinstance(author, list):
                german_ai_author_names.update(author)
            else:
                german_ai_author_names.add(author)
    with open(input_embeddings, 'r', encoding='utf-8') as in_f:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for line in tqdm(in_f):
                author = line.strip().split(";")[0]
                if author in german_ai_author_names:
                    out_f.write(line)


if __name__ == '__main__':
    for path in [["data/large_data/scibert_author_embeddings.csv", "data/german_author_scibert_embeddings.csv"]]:
        print("Doing path ", path[0])
        extract_german_ai_authors(path[0], "../Max/data/german_persons.json", path[1])

import ujson

from collections import defaultdict

from util import from_paper2author
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import non_negative_factorization, TruncatedSVD


def get_ai_authors(authors_file: str) -> set:
    authors = []
    with open(authors_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = ujson.loads(line)
            author = line['author'] if isinstance(line['author'], list) else [line['author']]
            authors.extend(author)
    return set(authors)


def de_list_item(item):
    if isinstance(item, list):
        if len(item) > 1:
            return ", ".join(item)
        elif len(item) == 1:
            return item[0]
        else:
            return ""
    return item


def get_content_string(pub: dict, keys: list = ('title', 'keywords', 'abstract')):
    content = ""
    for key in keys:
        if key in pub:
            content += de_list_item(pub[key])
    return content


def get_tfidf_matrix(input_path: str, years_remove: int = None):
    pubs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            pubs.append(ujson.loads(line))
    if years_remove:
        pubs = [x for x in pubs if int(x['year']) <= years_remove]
    texts = [(pub['key'], get_content_string(pub)) for pub in pubs]
    print("Got", len(texts), "texts. ")
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    tf_idf_matrix = vectorizer.fit_transform([x[1] for x in texts])
    return [x[0] for x in texts], tf_idf_matrix, vectorizer


def extract_topics(input_path: str, output_path: str = "data/large_data/", dimensions: int = 100,
                   years_remove: int = None):
    ids, tfidf_matrix, vectorizer = get_tfidf_matrix(input_path=input_path, years_remove=years_remove)
    print("TF-IDF shape is ", tfidf_matrix.shape)
    svd = TruncatedSVD(n_components=dimensions, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    print(type(reduced))
    print(reduced.shape)
    if years_remove == 2015:
        name_svd = "paper_svd_5.csv"
        name_matrix = "paper_matrix_5.csv"
    elif years_remove == 2018:
        name_svd = "paper_svd_2.csv"
        name_matrix = "paper_matrix_2.csv"
    else:
        name_svd = "paper_svd.csv"
        name_matrix = "paper_matrix.csv"
    with open(output_path + name_svd, 'w', encoding='utf-8') as f:
        for idx, vec in zip(ids, reduced):
            f.write(str(idx) + ";" + ",".join([str(x) for x in vec]) + "\n")
    w, h, n_iter = non_negative_factorization(tfidf_matrix, n_components=dimensions, init='random', random_state=42)
    print(type(w), type(h), n_iter)
    print(w.shape, h.shape)
    with open(output_path + name_matrix, 'w', encoding='utf-8') as f:
        for idx, vec in zip(ids, w):
            f.write(str(idx) + ";" + ",".join([str(x) for x in vec]) + "\n")


def concat_doc_approach(input_path: str, authors_file: str, type: str = "svd", dimensions: int = 100, output_path: str = "data/",
                        years_remove: int = None):
    """
    :param input_path:
    :param authors_file:
    :param type: Can be either svd or non neg matrix
    :param dimensions:
    :param output_path:
    :param years_remove:
    :return:
    """
    print("Starting with", type, "...")
    pubs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            pubs.append(ujson.loads(line))
    if years_remove:
        print("Removing from year", years_remove)
        pubs = [x for x in pubs if int(x['year']) >= years_remove]
    print("Having", len(pubs), "pubs. ")
    authors = [[(auth, x['key']) for auth in x['author']] for x in pubs]
    authors = [a for b in authors for a in b]
    ai_authors = get_ai_authors(authors_file=authors_file)
    authors = [auth for auth in authors if auth[0] in ai_authors]
    print("Length of authors after applying filters: ", len(authors))
    texts = {pub['key']: get_content_string(pub) for pub in pubs}
    author2text = defaultdict(str)
    print("And", len(authors), "authors. ")
    for author in authors:
        author2text[author[0]] += texts[author[1]] + " "
    id2author = {i: x for i, x in enumerate(author2text.keys())}
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    tf_idf_matrix = vectorizer.fit_transform(list(author2text.values()))
    if type == "svd":
        svd = TruncatedSVD(n_components=dimensions, random_state=42)
        reduced = svd.fit_transform(tf_idf_matrix)
        if years_remove == 2015:
            output_path += "german_author_singledoc_svd_5.csv"
        else:
            output_path += "german_author_singledoc_svd_2.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, vec in enumerate(reduced):
                f.write(str(id2author[i]) + ";" + ",".join([str(x) for x in vec]) + "\n")
    else:  # will be non neg matrix
        w, _, _ = non_negative_factorization(tf_idf_matrix, n_components=dimensions, init='random', random_state=42)
        if years_remove == "2015":
            output_path += "german_author_singledoc_matrix_5.csv"
        else:
            output_path += "german_author_singledoc_matrix_2.csv"
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, vec in enumerate(w):
                f.write(str(id2author[i]) + ";" + ",".join([str(x) for x in vec]) + "\n")


y2015 = 2015
y2018 = 2018

if __name__ == '__main__':
    # print("Concat approach of LSA for split 5 ...")
    # concat_doc_approach(input_path="data/large_data/german_ai_community_dataset.json",
    #                     authors_file="data/german_persons.json", years_remove=y2015)
    # print("Concat approach of Matrix for split 5 ...")
    # concat_doc_approach(input_path="data/large_data/german_ai_community_dataset.json", type="matrix",
    #                     authors_file="data/german_persons.json", years_remove=y2015)
    # print("Concat approach of LSA for split 2 ...")
    # concat_doc_approach(input_path="data/large_data/german_ai_community_dataset.json",
    #                     authors_file="data/german_persons.json", years_remove=y2018)
    # print("Concat approach of Matrix for split 2 ...")
    # concat_doc_approach(input_path="data/large_data/german_ai_community_dataset.json",
    #                     authors_file="data/german_persons.json", type="matrix", years_remove=y2018)
    # print("-"*100)
    # print("Extracting topics of split 2...")
    # extract_topics(input_path="data/large_data/german_ai_community_dataset.json", years_remove=y2018)
    # print("-"*100)
    # print("Extracting topics of split 5 ...")
    # extract_topics(input_path="data/large_data/german_ai_community_dataset.json", years_remove=y2015)
    print("Creating repr for each author ...")
    from_paper2author(input_paper="data/large_data/paper_svd_5.csv", output_path="data/german_author_svd_5.csv",
                      filter_authors=True)
    from_paper2author(input_paper="data/large_data/paper_svd_2.csv", output_path="data/german_author_svd_2.csv",
                      filter_authors=True)
    from_paper2author(input_paper="data/large_data/paper_matrix_5.csv", output_path="data/german_author_matrix_5.csv",
                      filter_authors=True)
    from_paper2author(input_paper="data/large_data/paper_matrix_2.csv", output_path="data/german_author_matrix_2.csv",
                      filter_authors=True)
    print("Finished. ")

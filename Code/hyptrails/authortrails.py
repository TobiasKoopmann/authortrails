import random
import shutil
import pyspark
import ujson
import sys

import numpy as np
from pathlib import Path
from typing import Tuple 
from collections import defaultdict
from itertools import groupby
from urllib.parse import urlparse
from scipy.sparse import csr_matrix, vstack

import geopy.distance as geo
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pytrails.sparktrails import matrixutils, MarkovChain
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

conf = SparkConf()
# conf.set('spark.driver.host', '127.0.0.1')
conf.set("spark.speculation", "false")
spark_context = SparkContext(conf=conf)
spark = SparkSession.builder.appName("REGIO Paper").config("spark.some.config.option", "some-value").getOrCreate()


# ----- Helpers, which needs to be pickeld -----
def de_list_item(item):
    if isinstance(item, list) and len(item) == 1:
        return item[0]
    elif isinstance(item, list) and len(item) > 1:
        return " ".join(item)
    elif isinstance(item, list) and len(item) == 0:
        return ""
    return item


def en_list_item(item):
    return item if isinstance(item, list) else [item]


def my_group(line: list) -> list:
    ret = []
    for val in groupby(line):
        ret.append((val[0], len(list(val[1]))))
    return ret


def list_to_sparse(line: list, size: int) -> csr_matrix:
    col = [int(x[0]) for x in line]
    row = np.zeros(len(col))
    data = [x[1] for x in line]
    return csr_matrix((data, (row, col)), shape=(1, size))


def sparse_to_string(sparse_row: csr_matrix) -> str:
    tmp = []
    for a, b in zip(sparse_row.indices, sparse_row.data):
        tmp.append(str(a) + ";" + str(b))
    return ",".join(tmp)


def get_top_k_similarity(author: str, author_paper: pyspark.broadcast, paper_vec: pyspark.broadcast, topk: int, vec_size: int) -> list:
    """
    :param author:
    :param author_paper: broadcast(dict(author_key: [paper_key]))
    :param paper_vec: broadcast(dict(paper_key: sparse_vec))
    :param topk:
    :param vec_size:
    :return:
    """
    sims = []
    if author not in author_paper.value:
        return []
    author_keys = author_paper.value[author]
    for auth, other_keys in author_paper.value.items():
        if author != auth:
            curr_vectors = [paper_vec.value[x] for x in author_keys if x not in other_keys]
            if len(curr_vectors) != 0:
                curr_vectors = vstack(curr_vectors).mean(axis=0)
            else:
                curr_vectors = csr_matrix(np.zeros(vec_size))
            other_vectors = [paper_vec.value[x] for x in other_keys if x not in author_keys]

            if len(other_vectors) != 0:
                other_vectors = vstack(other_vectors).mean(axis=0)
            else:
                other_vectors = csr_matrix(np.zeros(vec_size))
            sims.append((auth, cosine_similarity(curr_vectors, other_vectors)[0][0]))
    return sorted(sims, key=lambda x: x[1], reverse=True)[:topk]


def get_top_similarity_with_shared(author_id, author_paper: pyspark.broadcast, topk: int):
    sims = []
    curr_author_vec = author_paper.value[author_id]
    for auth, other_author in author_paper.value.items():
        if author_id != auth:
            sims.append((auth, cosine_similarity(curr_author_vec, other_author)[0][0]))
    return sorted([(x[0], x[1]) for x in sims], key=lambda x: x[1], reverse=True)[:topk]


def get_count_same_input(author: str, broadcast: pyspark.broadcast) -> list:
    """
    :param author: Name of the current author
    :param broadcast: A pyspark.broadcast with {author name: set of values}
    :return:
    """
    ret = []
    if author in broadcast.value:
        curr = broadcast.value[author]
        for auth, val in broadcast.value.items():
            if len(curr & val) != 0:
                ret.append((auth, len(curr & val)))
    return ret


# Geographic Helper
def get_geographic_closeness(author: str, broadcast: pyspark.broadcast) -> list:
    """
    :param author:
    :param broadcast:
    :return:
    """
    ids = []
    ret = []
    if author in broadcast.value:
        # curr is [(lat, long), ...]
        curr = broadcast.value[author]
        for auth, val in broadcast.value.items():
            # val is [(lat, long), ...]
            distances = [geo.great_circle(a, b).km for a in curr for b in val]
            ids.append(auth)
            ret.append(min(distances))
    if len(ret) != 0:
        ret = normalize(np.array(ret).reshape(1, -1))[0]
        return [(auth, 1 - val) for auth, val in zip(ids, ret)]
    return []


# Web Distance Helper
def flatten_web_distance_line(line: tuple) -> list:
    if ";" in line[0]:
        x_names = line[0].split(";")
    else:
        x_names = [line[0]]
    if ";" in line[1]:
        y_names = line[1].split(";")
    else:
        y_names = [line[1]]
    if ";" in line[2]:
        dist = line[2].split(";")
    else:
        dist = [line[2]]
    return [(x.strip(), y.strip(), int(d.strip())) for x in x_names for y in y_names for d in dist]


def web_distance_grouping(line: list) -> list:
    distances = defaultdict(list)
    for element in line:
        distances[element[0]].append(int(element[1]))
    ids = distances.keys()
    values = normalize(np.array([float(sum(x)) / len(x) for x in distances.values()]).reshape(1, -1))[0]
    return [(auth, 1 - val) for auth, val in zip(ids, values)]


def get_same_institutional(author: int, uni: pyspark.broadcast, private: pyspark.broadcast):
    res = []
    if author in uni.value:
        res.extend([x for x in uni.value if x != author])
    if author in private.value:
        res.extend([x for x in private.value if x != author])
    return set(res)


class SparkHypothesis:
    def __init__(self, 
                 hypothesis: list, 
                 base_dir: Path = Path("data"),
                 pub_file_name: Path = Path("german_ai_community_dataset.json"), 
                 gaw_seeds: Path = Path("gaw.seeds"),
                 hyp_dir: Path = Path("authortrails"), 
                 institutional_split: Path = Path("private_vs_uni.csv"), 
                 geo_data_path: Path = Path("data_Distance.csv"), 
                 geo_data_city_path: Path = Path("cities.csv"),
                 dnb_file: Path = Path("DNB_LINK2020-05-07.csv"),
                 author_information_path: str = "german_persons.json", 
                 author_key: str = "author", 
                 calculate_hypothesis: bool = True,
                 calculate_evidence: bool = True, 
                 overwrite_evidence: bool = True, 
                 year_split: str = None,
                 topk: int = 10,
                 load_partitions: int = 500,
                 save_partitions: int = 2,
                 verbose: int = 2):
        print("DEBUG: Running Spark Hypothesis in ", base_dir, " on file ", pub_file_name, ".")
        self.verbose = verbose
        self.load_partitions = load_partitions
        self.save_partitions = save_partitions
        self.topk = topk
        self.author_key = author_key
        self.hypothesis = hypothesis
        self.basedir = base_dir
        if year_split:
            self.hyp_dir = Path(self.basedir, f"{hyp_dir}_{year_split}")
            file_ending = f"_{year_split}.csv"
        else:
            self.hyp_dir = Path(self.basedir, hyp_dir)
            file_ending = ".csv"
        if year_split == "5":
            self.dataset = "authortrails"
            self.year_split = "5"
            self.years = {"2015", "2016", "2017", "2018", "2019"}
        elif year_split == "2":
            self.dataset = "authortrails"
            self.year_split = "2"
            self.years = {"2018", "2019"}
        else:
            self.dataset = "pattrails"
            self.year_split = "all"
            self.years = {str(x) for x in np.arange(1970, 2020)}
        self.latent_switcher = {
            "bert": Path(self.basedir, f"german_author_scibert_embeddings{file_ending}"),
            "svd": Path(self.basedir, f"german_author_svd{file_ending}"),
            "svd_singleDoc": Path(self.basedir, f"german_author_singledoc_svd{file_ending}"),
            "matrix": Path(self.basedir, f"german_author_matrix{file_ending}"),
            "matrix_singleDoc": Path(self.basedir, f"german_author_singledoc_matrix{file_ending}"),
            "deepwalk": Path(self.basedir, f"deepwalk{file_ending}"),
            "node2vec_largeP": Path(self.basedir, f"node2vec_largeP{file_ending}"),
            "node2vec_smallP": Path(self.basedir, f"node2vec_smallP{file_ending}"),
            "hope": Path(self.basedir, f"hope{file_ending}")
            }
        self.distance_switcher = {
            "syntactic_web": Path(self.basedir, "synt_interaction.tsv"),
            "link_distance": Path(self.basedir, "interHost_interaction.tsv")
        }
        self.dict_path = Path(self.basedir, author_information_path)
        self.publication_file = Path(self.basedir, pub_file_name)
        self.gaw_seeds = Path(self.basedir, gaw_seeds)
        self.geo_data_path = Path(self.basedir, geo_data_path)
        self.geo_data_city_path = Path(self.basedir, geo_data_city_path)
        self.dnb_file = Path(self.basedir, dnb_file)
        self.institutional_split = Path(self.basedir, institutional_split)
        publication, _, self.amount_inventors = self.load_publication_and_author_id_dict()
        self.amount_publication = publication.count()
        self.overwrite_evidence = overwrite_evidence
        if calculate_hypothesis:
            self.run_hypothesis()
        if calculate_evidence:
            self.run_evidence()

    # --- Helpers ----
    @staticmethod
    def string_to_sparse(hyp_string: str, max_id: int) -> Tuple[str, csr_matrix]:
        index = int(hyp_string.split("\t")[0])
        data = hyp_string.split("\t")[1].split(",")
        indecies = [int(x.split(";")[0]) for x in data]
        values = [float(x.split(";")[1]) for x in data]
        return index, csr_matrix((values, indecies, [0, len(values)]), shape=(1, max_id))

    # --- Method ---
    def load_affiliations(self, authors: pyspark.broadcast) -> list:
        affiliations = []
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = ujson.loads(line)
                if 'note' in line:
                    affiliations.append((en_list_item(line[self.author_key]), en_list_item(line['note'])))
        affiliations = [a for b in [[(auth, set([institution.split(",")[0] for institution in x[1]]))
                                     for auth in x[0]] for x in affiliations] for a in b]
        return [(authors.value[x[0]], x[1]) for x in affiliations if x[0] in authors.value]

    # --- Method ---
    @staticmethod
    def match_affiliations(affiliations: set, geo_locs: list) -> dict:
        returns = dict()
        for geo_loc in geo_locs:
            geo_name = geo_loc[0].split("(")[0].strip()
            if geo_name in affiliations:
                returns[geo_name] = float(geo_loc[2]), float(geo_loc[3])
            elif geo_name.replace("of ", "") in affiliations:
                returns[geo_name.replace("of ", "")] = float(geo_loc[2]), float(geo_loc[3])
            elif geo_name.replace("of ", "").replace("University", "Universität") in affiliations:
                returns[geo_name("of ", "").replace("University", "Universität")] = \
                    float(geo_loc[2]), float(geo_loc[3])
            if geo_name == "Ludwig-Maximilians-Universität München":
                returns["LMU München"] = float(geo_loc[2]), float(geo_loc[3])
            elif geo_name == "Julius-Maximilians-Universität Würzburg":
                returns["University of Würzburg"] = float(geo_loc[2]), float(geo_loc[3])
                returns["Julius Maximilians University of Würzburg"] = float(geo_loc[2]), float(geo_loc[3])
                returns["Julius Maximilian University of Würzburg"] = float(geo_loc[2]), float(geo_loc[3])
                returns["Universität Würzburg"] = float(geo_loc[2]), float(geo_loc[3])
        return returns

    # --- Method ---
    def load_publication_and_author_id_dict(self, load_test: bool = False) -> Tuple[pyspark.rdd, pyspark.broadcast, int]:
        # For spark issues, class properties can not be used on workers
        author_key = self.author_key
        publication = spark_context.textFile(self.publication_file, self.load_partitions).map(lambda x: ujson.loads(x))
        if self.dataset == "authortrails":
            years_bc = spark_context.broadcast(self.years)
            publication_pre = publication.filter(lambda x: x['year'] in years_bc.value)
            if load_test:
                publication = publication_pre
            else:
                publication = publication.filter(lambda x: x['year'] not in years_bc.value)
        else:
            publication_pre = publication
        authors_in_pubs = set(publication_pre.map(lambda x: x[author_key]).flatMap(lambda x: x).distinct().collect())
        authors = spark_context.textFile(self.dict_path, self.load_partitions).map(lambda x: ujson.loads(x))
        authors = [en_list_item(x[author_key]) for x in authors.collect()]
        authors = [x for x in authors if any(auth in x for auth in authors_in_pubs)]
        authors = [a for b in [[(auth, i) for auth in x] for i, x in enumerate(authors)] for a in b]
        authors_count = len(set([x[1] for x in authors]))
        print("\tDEBUG: Having", authors_count, "authors after filtering. ")
        return publication, spark_context.broadcast({x[0]: x[1] for x in authors}), authors_count

    # --- Method ---
    def save_hypothesis(self, hypothesis: pyspark.rdd, output_path: str, parse_from_list: bool = True,
                        parse_from_sparse: bool = False) -> None:
        """
        Checks if already exists, if yes, deletes and saves
        :param self:
        :param hypothesis: What to save
        :param output_path: Where to save
        :param parse_from_list:
        :param parse_from_sparse:
        :return: None
        """
        # For spark issues, class properties can not be used on workers
        amount_authors = self.amount_inventors
        save_partitions = self.save_partitions
        if self.verbose == 2:
            print("\tDEBUG: Saving hypothesis. ")
        if parse_from_list:
            hypothesis = hypothesis.map(lambda x: (x[0], list_to_sparse(x[1], size=amount_authors)))
            parse_from_sparse = True
        if Path.exists(output_path):
            shutil.rmtree(output_path)
        if parse_from_sparse:
            hypothesis = hypothesis.map(lambda x: (str(x[0]) + "\t" + sparse_to_string(x[1])))
            hypothesis = hypothesis.filter(lambda x: len(x.split("\t")[1]) != 0)
        hypothesis.coalesce(save_partitions).saveAsTextFile(output_path)

    # --- Hypothesis ---
    def create_data_hypothesis(self, output_path=None) -> pyspark.rdd:
        """
        The file needs to be saved like this in order to be read by the pytrails lib:
        ID \t [Target;count]
        :param self:
        :param output_path: Where to save, if None, will not save
        :return:
        """
        # For spark issues, class properties can not be used on workers
        author_key = self.author_key
        publication, authors, _ = self.load_publication_and_author_id_dict()
        hyp_true = publication.map(lambda x: x[author_key]). \
            map(lambda x: [authors.value[auth] for auth in x if auth in authors.value]). \
            map(lambda x: [[x[j], x[k]] for j in range(len(x)) for k in range(len(x)) if j != k]). \
            flatMap(lambda x: x).groupByKey().map(lambda x: (x[0], my_group(x[1])))
        if self.verbose == 2:
            print("\tDEBUG: Having ", hyp_true.count(), "authors. ")
        if output_path is not None:
            self.save_hypothesis(hypothesis=hyp_true, output_path=output_path + "hyp_true")
        return hyp_true

    # --- Method ---
    def create_self_hypothesis(self, output_path=None) -> pyspark.rdd:
        """
        :param self:
        :param output_path: if None, will not save
        :return:
        """
        # Loading the file
        publication, authors, _ = self.load_publication_and_author_id_dict()
        hyp_self = spark_context.parallelize(authors.value.values()).distinct() \
            .map(lambda x: str(x) + "\t" + str(x) + ";1.0")
        if self.verbose == 2:
            print("\tDEBUG: Saving self hypothesis. ")
        if output_path is not None:
            self.save_hypothesis(hypothesis=hyp_self, output_path=output_path + "hyp_self", parse_from_list=False)
        return hyp_self

    # --- Method ---
    def create_random_hypothesis(self, output_path=None) -> pyspark.rdd:
        """
        :param self:
        :param output_path: if None, will not save
        :return:
        """
        author_key = self.author_key
        # Loading the file
        publication, authors, _ = self.load_publication_and_author_id_dict()
        publication = publication.map(lambda x: x[author_key]). \
            map(lambda x: [authors.value[auth] for auth in x if auth in authors.value]). \
            map(lambda x: [[auth] + [random.choice(list(authors.value.values())) for _ in range(len(x) - 1)]
                           for auth in x]).flatMap(lambda x: x)
        random_hyp = publication.map(lambda x: [[x[j], x[k]] for j in range(len(x)) for k in range(len(x)) if j != k]). \
            flatMap(lambda x: x).groupByKey().map(lambda x: (x[0], my_group(x[1])))
        if self.verbose == 2:
            print("\tDEBUG: Having ", random_hyp.count(), "authors. ")
        if output_path is not None:
            self.save_hypothesis(hypothesis=random_hyp, output_path=output_path + "hyp_random_coauthor")
        return random_hyp

    # --- Method ---
    def create_latent_author_representation_hypothesis(self, rep_type: str, output_path: str = None, topk: int = 50):
        """
        :param output_path:
        :param rep_type:
        :param topk:
        :return:
        """
        # For spark issues, class properties can not be used on workers
        load_partitions = self.load_partitions
        representation_path = self.latent_switcher.get(rep_type)
        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for latent paper representation hypothesis. ")
        # Loading the file
        pubs, authors, _ = self.load_publication_and_author_id_dict()
        latent_rep = spark_context.textFile(representation_path, load_partitions).map(lambda x: x.split(";")). \
            filter(lambda x: x[0] in authors.value). \
            map(lambda x: (authors.value[x[0]], csr_matrix([float(y) for y in x[1].split(",")])))
        vec_size = latent_rep.first()[-1].shape[-1]
        if self.verbose == 2:
            print("\tDEBUG: Having a latent representation with ", vec_size, "entries. ")
        latent_rep = spark_context.broadcast({x[0]: x[1] for x in latent_rep.collect()})
        hyp_lat = spark_context.parallelize(authors.value.values(), load_partitions).distinct(). \
            filter(lambda x: x in latent_rep.value). \
            map(lambda x: (x, get_top_similarity_with_shared(author_id=x, author_paper=latent_rep, topk=topk))). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(y[0]) + ";" + str(y[1] * 1) for y in x[1]]))
        if output_path is not None:
            self.save_hypothesis(hypothesis=hyp_lat, output_path=output_path + "hyp_lat_" + rep_type,
                                 parse_from_list=False)
        return hyp_lat

    # --- Method ---
    def create_affiliation_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for Affiliation hypothesis. ")
        # Loading the file
        _, authors, _ = self.load_publication_and_author_id_dict()
        affiliations = spark_context.broadcast({x[0]: x[1] for x in self.load_affiliations(authors=authors)})
        hyp_affiliation = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_count_same_input(author=x, broadcast=affiliations))). \
            filter(lambda x: x[1] and len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_affiliation, output_path=output_path + "hyp_affiliation",
                                 parse_from_list=False)
        return hyp_affiliation

    # --- Method ---
    def create_geo_dist_of_affiliation_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for geographic distance of affiliation hypothesis. ")
        # Loading the file
        _, authors, _ = self.load_publication_and_author_id_dict()
        author_affiliation = self.load_affiliations(authors=authors)
        affiliations = set().union(*[x[1] for x in author_affiliation])
        with open(self.geo_data_path, 'r', encoding="utf-8") as f:
            geo_locations = f.readlines()
        geo_locations = self.match_affiliations(affiliations, [x.split(";") for x in geo_locations[1:]])
        author_affiliation = [(x[0], [geo_locations[y] for y in x[1] if y in geo_locations]) for x in
                              author_affiliation]
        author_affiliation = spark_context.broadcast({x[0]: x[1] for x in author_affiliation if len(x[1]) > 0})
        hyp_geo_affiliation = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_geographic_closeness(author=x, broadcast=author_affiliation))). \
            filter(lambda x: x[1] and len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_geo_affiliation, output_path=output_path + "hyp_geo_affiliation",
                                 parse_from_list=False)
        return hyp_geo_affiliation

    # --- Method ---
    def create_geo_dist_of_city_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """

        def match_city(affiliation_set: set, geo_locs: dict) -> list:
            ret_list = []
            for affiliation in affiliation_set:
                matched = False
                for name, coords in geo_locs.items():
                    if name in affiliation and not matched:
                        ret_list.append(coords)
                        matched = True
            return ret_list

        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for geographic distance of affiliation hypothesis. ")
        # Loading the file
        _, authors, _ = self.load_publication_and_author_id_dict()
        author_affiliation = self.load_affiliations(authors=authors)
        with open(self.geo_data_city_path, 'r', encoding="utf-8") as f:
            geo_locations = f.readlines()
        cities = dict()
        for location in [x.split(",") for x in geo_locations[1:]]:
            cities[location[1]] = float(location[3]), float(location[4])
            cities[location[2]] = float(location[3]), float(location[4])
        author_affiliation = [(x[0], match_city(x[1], cities)) for x in author_affiliation]
        author_affiliation = spark_context.broadcast({x[0]: x[1] for x in author_affiliation if len(x[1]) != 0})
        hyp_geo_city = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_geographic_closeness(author=x, broadcast=author_affiliation))). \
            filter(lambda x: x[1] and len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_geo_city, output_path=output_path + "hyp_geo_city",
                                 parse_from_list=False)
        return hyp_geo_city

    # --- Method ---
    def create_same_dissertation_location_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        dnb_file = self.dnb_file
        load_partitions = self.load_partitions
        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for geographic distance of PhD-affiliation hypothesis. ")
        # Loading the file
        _, authors, _ = self.load_publication_and_author_id_dict()
        dnb = spark_context.textFile(dnb_file, load_partitions).map(lambda x: x.strip().split(";")). \
            map(lambda x: (x[2].replace('\"', ''), x[6].replace('\"', ''))).filter(lambda x: x[0] != "name")
        if self.verbose == 2:
            print("\tDEBUG: Having", dnb.count(), "data points before mapping to authors dict.")
        dnb = dnb.filter(lambda x: x[0] in authors.value).map(lambda x: (authors.value[x[0]], x[1]))
        if self.verbose == 2:
            print("\tDEBUG: Having", dnb.count(), "data points.")
        dnb_bc = spark_context.broadcast({x[0]: {x[1]} for x in dnb.collect()})
        hyp_diss_location = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_count_same_input(author=x, broadcast=dnb_bc))). \
            filter(lambda x: x[1] and len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_diss_location, output_path=output_path + "hyp_diss_loc",
                                 parse_from_list=False)
        return hyp_diss_location

    # --- Method ---
    def create_same_conference_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        author_key = self.author_key
        pubs, authors, _ = self.load_publication_and_author_id_dict()
        auth_venue = pubs.map(
            lambda x: [(auth, de_list_item(x['booktitle']) + de_list_item(x['year'])) if x['booktitle'] else
                       (auth, de_list_item(x['journal']) + de_list_item(x['year'])) for auth in x[author_key]]). \
            flatMap(lambda x: x).filter(lambda x: x[0] in authors.value). \
            map(lambda x: (authors.value[x[0]], x[1])).distinct()
        venue_visits = spark_context.broadcast({x[0]: x[1] for x in auth_venue.map(lambda x: (x[1], x[0])).groupByKey().
                                               map(lambda x: (x[0], [x for x in x[1]])).collect()})
        # Generating the sparse matrix
        hyp_venue = auth_venue.groupByKey().map(lambda x: (x[0], [venue_visits.value[x] for x in x[1]])). \
            map(lambda x: (x[0], [a for b in x[1] for a in b])).map(lambda x: (x[0], my_group(x[1]))). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if output_path is not None:
            self.save_hypothesis(hypothesis=hyp_venue, output_path=output_path + "hyp_conference",
                                 parse_from_list=False)
        return hyp_venue

    # --- Method ---
    def create_same_url_host_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        dict_path = self.dict_path
        load_partitions = self.load_partitions
        if self.verbose == 2:
            print("\tDEBUG: Loading pubs for URL hypothesis. ")
        _, authors, _ = self.load_publication_and_author_id_dict()
        seeds = set()
        with open(self.gaw_seeds, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('+'):
                    seeds.add(line.strip().replace('+', ''))
        author_url = spark_context.textFile(dict_path, load_partitions). \
            map(lambda x: ujson.loads(x)).filter(lambda x: "url" in x). \
            map(lambda x: (x['author'], en_list_item(x['url']))). \
            map(lambda x: (x[0], [".".join(urlparse(url).netloc.split(".")[-2:]) for url in x[1]])). \
            map(lambda x: (x[0], set([x for x in x[1] if x in seeds])))
        author_url = [a for b in [[(auth, x[1]) for auth in x[0]] if isinstance(x[0], list) else [x]
                                  for x in author_url.collect()] for a in b]
        author_url = spark_context.broadcast({authors.value[x[0]]: x[1] for x in author_url if x[0] in authors.value})
        hyp_url = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_count_same_input(author=x, broadcast=author_url))). \
            filter(lambda x: x[1] and len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_url, output_path=output_path + "hyp_url", parse_from_list=False)
        return hyp_url

    # --- Method ---
    def create_web_distance_hypothesis(self, distance: str, output_path: str = None):
        """
        :param output_path:
        :param distance:
        :return:
        """
        distance_path = self.distance_switcher[distance]
        load_partitions = self.load_partitions
        _, authors, _ = self.load_publication_and_author_id_dict()
        distances = spark_context.textFile(distance_path, load_partitions).map(lambda x: x.strip().split("\t"))
        hyp_distances = distances.map(lambda x: flatten_web_distance_line(x)).flatMap(lambda x: x). \
            filter(lambda x: x[0] in authors.value and x[1] in authors.value). \
            map(lambda x: (authors.value[x[0]], (authors.value[x[1]], x[2])))
        if self.verbose == 2:
            print("\tDEBUG: Having", hyp_distances.count(), "datapoints. ")
        hyp_distances = hyp_distances.groupByKey().map(lambda x: (x[0], web_distance_grouping(x[1])))
        hyp_distances = hyp_distances. \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(val[0]) + ";" + str(val[1] * 1) for val in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_distances, output_path=output_path + "hyp_" + distance,
                                 parse_from_list=False)
        return hyp_distances

    # --- Method ---
    def create_institutional_hypothesis(self, output_path: str = None):
        """
        :param output_path:
        :return:
        """
        institutional_split = self.institutional_split
        _, authors, _ = self.load_publication_and_author_id_dict()
        dict_path = self.dict_path
        key2name = spark_context.textFile(dict_path).map(lambda x: ujson.loads(x)). \
            map(lambda x: (x['key'], de_list_item(x['author']))).filter(lambda x: x[1] in authors.value)
        key2name = {x[0]: x[1] for x in key2name.collect()}
        if self.verbose:
            print("\tDEBUG: Len of key2name", len(key2name), ".")
        key2name = spark_context.broadcast(key2name)
        institutions = spark.read.csv(institutional_split, header=True).select("key", "uni?").rdd
        if self.verbose:
            print("\tDEBUG: Institutional: ", institutions.take(5))
        institutions = institutions.filter(lambda x: x['key'] in key2name.value)
        if self.verbose:
            print("\tDEBUG: Len of institutions", institutions.count(), ".")
        uni = spark_context.broadcast(set(institutions.filter(lambda x: x["uni?"] == "True").map(
            lambda x: authors.value[key2name.value[x['key']]]).collect()))
        private = spark_context.broadcast(set(institutions.filter(lambda x: x["uni?"] == "False").map(
            lambda x: authors.value[key2name.value[x['key']]]).collect()))
        hyp_institution = spark_context.parallelize(authors.value.values()).distinct(). \
            map(lambda x: (x, get_same_institutional(x, uni, private)))
        if self.verbose:
            print("\tDEBUG: First few lines of hypothesis; ", hyp_institution.take(5))
        hyp_institution = hyp_institution.filter(lambda x: len(x[1]) > 0). \
            map(lambda x: str(x[0]) + "\t" + ",".join([str(auth) + ";1.0" for auth in x[1]]))
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path:
            self.save_hypothesis(hypothesis=hyp_institution, output_path=output_path + "hyp_institutional",
                                 parse_from_list=False)
        return hyp_institution

    # --- Method ---
    @staticmethod
    def save_evidence_dict(output_dict: dict, output_path: str):
        """
        :param output_dict:
        :param output_path:
        :return:
        """
        with open(output_path + "evidence.json", "w", encoding="utf-8") as f:
            ujson.dump(output_dict, f)

    # --- Method ---
    def calc_evidence_from_files(self, hypothesis: list, output_path: str = None,
                                 k_values: list = (0, 1, 2, 3, 4, 5, 6)) -> dict:
        """
        :param self:
        :param hypothesis: list of hypothesis, e.g. ["true", "self"]
        :param output_path: Where to save the output dict, not saving if None
        :param k_values:
        :return:
        """
        # For spark issues, class properties can not be used on workers
        working_dir = self.hyp_dir
        amount_authors = self.amount_inventors
        if self.verbose == 2:
            print("\tDEBUG: Loading data hypothesis. ")
        hyp_true = matrixutils.textfile_rdd_to_csr_matrix_rdd(spark_context.textFile(working_dir + "hyp_true"),
                                                              number_of_columns=amount_authors)
        return_dict = {"max_id": amount_authors, "k_values": k_values}
        ks = (np.array(k_values) * amount_authors).astype(int)
        for hypo in hypothesis:
            if isinstance(hypo, list):
                hyp_name = "_".join(hypo)
            else:
                hyp_name = hypo
            if self.verbose:
                print("DEBUG: Loading another hypothesis:", hyp_name)
            current_hyp = matrixutils.textfile_rdd_to_csr_matrix_rdd(
                spark_context.textFile(working_dir + "hyp_" + hyp_name),
                number_of_columns=amount_authors). \
                mapValues(lambda line: normalize(line, "l1", axis=1))
            current_hyp.map(lambda x: x[1].shape).distinct().collect()
            amount_data_points = current_hyp.count()
            if self.verbose:
                print("DEBUG: Calculating marginal likelihood for", hyp_name)
            return_dict[hyp_name] = \
                {"evidence_values": list(MarkovChain.marginal_likelihood(hyp_true, current_hyp, ks, 1.0)),
                 "data_points": amount_data_points}
        if self.verbose == 2:
            print("\tDEBUG: Saving to", output_path)
        if output_path is not None:
            self.save_evidence_dict(output_dict=return_dict, output_path=output_path)
        print("DEBUG: Finished. ")
        return return_dict

    # --- Method ---
    def run_hypothesis(self) -> None:
        # For spark issues, class properties can not be used on workers
        working_dir = self.hyp_dir
        hypothesis_to_calc = self.hypothesis.copy()
        if self.verbose:
            print("DEBUG: Calculating all the hypothesis. ", hypothesis_to_calc)

        if "syntactic_web" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating web distance based on syntactic depth hypothesis. ")
            self.create_web_distance_hypothesis(distance="syntactic_web", output_path=working_dir)
            hypothesis_to_calc.remove("syntactic_web")

        if "link_distance" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating link distance on same hosts hypothesis. ")
            self.create_web_distance_hypothesis(distance="link_distance", output_path=working_dir)
            hypothesis_to_calc.remove("link_distance")

        if "diss_loc" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating same Dissertation location hypothesis. ")
            self.create_same_dissertation_location_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("diss_loc")

        if "geo_city" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating geographic city hypothesis. ")
            self.create_geo_dist_of_city_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("geo_city")

        if "geo_affiliation" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating geographic  affiliation hypothesis. ")
            self.create_geo_dist_of_affiliation_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("geo_affiliation")

        if "true" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating data hypothesis. ")
            self.create_data_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("true")

        if "self" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating self hypothesis. ")
            self.create_self_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("self")

        if "random_coauthor" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating random_coauthor hypothesis. ")
            self.create_random_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("random_coauthor")

        if "conference" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Conference hypothesis. ")
            self.create_same_conference_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("conference")

        if "affiliation" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Affiliation hypothesis. ")
            self.create_affiliation_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("affiliation")

        if "url" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating URL hypothesis. ")
            self.create_same_url_host_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("url")

        if "lat_bert" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Bert hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="bert")
            hypothesis_to_calc.remove("lat_bert")

        if "lat_svd" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating SVD hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="svd")
            hypothesis_to_calc.remove("lat_svd")

        if "lat_svd_singleDoc" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating SVD hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="svd_singleDoc")
            hypothesis_to_calc.remove("lat_svd_singleDoc")

        if "lat_matrix" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating non Neg Matrix hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="matrix")
            hypothesis_to_calc.remove("lat_matrix")

        if "lat_matrix_singleDoc" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating non Neg Matrix hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="matrix_singleDoc")
            hypothesis_to_calc.remove("lat_matrix_singleDoc")

        if "lat_deepwalk" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Deepwalk hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="deepwalk")
            hypothesis_to_calc.remove("lat_deepwalk")

        if "lat_node2vec_largeP" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Node2vec with large P hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="node2vec_largeP")
            hypothesis_to_calc.remove("lat_node2vec_largeP")

        if "lat_node2vec_smallP" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Node2vec with small P hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="node2vec_smallP")
            hypothesis_to_calc.remove("lat_node2vec_smallP")

        if "lat_hope" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating Hope hypothesis. ")
            self.create_latent_author_representation_hypothesis(output_path=working_dir, rep_type="hope")
            hypothesis_to_calc.remove("lat_hope")

        if "institutional" in hypothesis_to_calc:
            if self.verbose:
                print("DEBUG: Calculating institutional hypothesis. ")
            self.create_institutional_hypothesis(output_path=working_dir)
            hypothesis_to_calc.remove("institutional")

    # --- Method ---
    def run_evidence(self):
        if self.verbose:
            print("DEBUG: Calculating evidence. ")
        working_dir = self.hyp_dir
        hypothesis = self.hypothesis
        self.calc_evidence_from_files(output_path=working_dir, hypothesis=hypothesis)
        self.calc_evidence_from_files(output_path=working_dir,
                                      k_values=[0, 1, 3, 5, 10, 100, 1000, 5000, 10000, 1000000],
                                      hypothesis=hypothesis)


if __name__ == '__main__':
    SparkHypothesis(hypothesis=["true", "self", "random_coauthor", "affiliation", "geo_affiliation", "geo_city",
                                "conference", "url", "lat_bert", "lat_svd", "lat_matrix", "lat_svd_singleDoc",
                                "lat_matrix_singleDoc", "lat_deepwalk", "lat_node2vec_largeP", "institutional",
                                "lat_node2vec_smallP", "lat_hope", "diss_loc", "syntactic_web", "link_distance"]
    )

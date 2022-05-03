import ujson
import csv

from tqdm import tqdm
from typing import Tuple
import networkx as nx


def create_graph(pat_data: list):
    links = []
    for pat in tqdm(pat_data):
        ids = list(pat['cp_ids'])
        if len(ids) > 1:
            links.extend([(ids[i], ids[j]) for i in range(len(ids)) for j in range(len(ids)) if i < j])
    graph = nx.Graph()
    for link in links:
        graph.add_edge(link[0], link[1])
    graphs = list(nx.connected_components(graph))
    print('Nodes:', len(graph))
    print('Subgraph:', len(max(graphs, key=len)))


def loading_patstat_data(input_file: str) -> Tuple[dict, dict, dict, set]:
    pub_data, pat_data, co_inventor_data, inventors = {}, {}, {}, set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=3760478):
            line = line.strip().replace("\"", "").split(";")
            if line[1] != "publication_id":
                if line[1] in pub_data:  # should be publication_id
                    pub_data[line[1]]["dblp_names"].update([line[2], line[22]])
                    pub_data[line[1]]["dblp_ids"].add(line[8])
                else:
                    pub_data[line[1]] = {"publication_id": line[1],
                                         "dblp_names": {line[2], line[22]},
                                         "dblp_ids": {line[8]}}
                if line[4] in pat_data:  # should be appln_id
                    pat_data[line[4]]["cp_names"].add(line[12])
                    pat_data[line[4]]["cp_ids"].update([line[3], line[5]])
                else:
                    pat_data[line[4]] = {"appln_id": line[4],
                                         "cp_names": {line[12]},
                                         "cp_ids": {line[3], line[5]},
                                         }
                if line[3] in co_inventor_data:
                    co_inventor_data[line[3]]["country"].add(line[11])
                    co_inventor_data[line[3]]["address1"].add(line[13])
                    co_inventor_data[line[3]]["address2"].add(line[14])
                    co_inventor_data[line[3]]["city"].add(line[15])
                    co_inventor_data[line[3]]['county'].add(line[16])
                    co_inventor_data[line[3]]['region'].add(line[17])
                    co_inventor_data[line[3]]['state'].add(line[18])
                    co_inventor_data[line[3]]['zip'].add(line[19])
                    co_inventor_data[line[3]]['dblp_id'].add(line[8])
                    co_inventor_data[line[3]]['dblp_name'].add(line[22])
                else:
                    co_inventor_data[line[3]] = {
                        "cp_id": line[3],
                        "country": {line[11]},
                        "address1": {line[13]},
                        "address2": {line[14]},
                        "city": {line[15]},
                        "county": {line[16]},
                        "region": {line[17]},
                        "state": {line[18]},
                        "zip": {line[19]},
                        "dblp_id": {line[8]},
                        "dblp_name": {line[22]}
                    }
                    inventors.add(line[5])
    return pub_data, pat_data, co_inventor_data, inventors


def un_set_lists(current_dict: dict):
    for k, v in current_dict.items():
        if isinstance(v, set):
            current_dict[k] = list(v)
    return current_dict


def un_list_values(current_dict: dict):
    for k, v in current_dict.items():
        if isinstance(v, list):
            current_dict[k] = v[0]
    return current_dict


def load_patent_data(input_path: str, output_dir: str):
    pub_data, pat_data, co_inventor_data, inventors = loading_patstat_data(input_path)
    pat_keys = list(pat_data.keys())
    print(len(pat_keys))
    print('Pats:')
    for i in pat_keys[:5]:
        print(pat_data[i])
    print()
    print('Co-Inventors:', len(co_inventor_data))
    print('Inventors: ', len(inventors))
    for i in list(inventors)[:5]:
        print(co_inventor_data[i])
    create_graph(list(pat_data.values()))
    with open(output_dir + "patents.json", 'w', encoding='utf-8') as f:
        for line in pat_data.values():
            ujson.dump(un_set_lists(line), f)
            f.write("\n")
    with open(output_dir + "dblp_publications.json", 'w', encoding='utf-8') as f:
        for line in pub_data.values():
            ujson.dump(un_set_lists(line), f)
            f.write("\n")
    with open(output_dir + "inventors.json", 'w', encoding='utf-8') as f:
        for line in co_inventor_data.values():
            ujson.dump(un_list_values(un_set_lists(line)), f)
            f.write("\n")


# --- For Patent description ---
def load_patent_content(input_path: str = "./data/patent-content.csv"):
    patents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        curr_patent = {}
        has_patent = False
        f.readline()  # skip the first line
        line = f.readline()
        while line:
            line = line.strip().split("\";\"")
            line = line[0].replace("\"", "").split(";") + line[1:]
            if has_patent and len(line) == 5:
                patents.append(curr_patent)
                curr_patent = {}
            if len(line) == 5:
                curr_patent['id'] = line[0].replace("\"", "")
                curr_patent['appln_id'] = line[1].replace("\"", "")
                curr_patent['APDT'] = line[2].replace("\"", "")
                curr_patent['title'] = line[3].replace("\"", "")
                curr_patent['abstract'] = line[4].replace("\"", "")
                has_patent = True
            else:
                if len(line) == 1:
                    curr_patent['abstract'] += line[0].replace("\"", "")
                elif len(line) == 2:
                    curr_patent['title'] += line[0].replace("\"", "")
                    curr_patent['abstract'] = line[1].replace("\"", "")
            line = f.readline()
    print("Length:", len(patents))
    return patents


# --- If Inventions file available
def from_inventions2patents(input_path: str = "./data/pattrails/Inventions2020-05-13.csv",
                            output_path: str = "./data/pattrails/patents.json"):
    patent_list = []
    with open(input_path, 'r', encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";", quotechar="\"")
        for row in r:
            patent_list.append(row)
    patent_dict = dict()
    for pat in tqdm(patent_list[1:]):
        pat_id = int(pat[1])
        if pat_id not in patent_dict:
            patent_dict[pat_id] = {"author_keys": {pat[2]}, "author": {pat[3]},
                                   "patstat_ids": {int(pat[4])}, "patstat_names": {pat[5]}}
        else:
            curr = patent_dict[int(pat[1])]
            curr["author_keys"].add(pat[2])
            curr["author"].add(pat[3])
            curr["patstat_ids"].add(int(pat[4]))
            curr["patstat_names"].add(pat[5])
            patent_dict[pat_id] = curr
    tmp = [x for x in patent_dict.values() if len(x['author']) > 1]
    co_inventor_ships = [[(a, b) for a in val['author'] for b in val['author'] if a != b]
                         for val in tmp]
    co_inventor_ships = set([a for b in co_inventor_ships for a in b])
    print(list(co_inventor_ships)[:5])
    print("Co-inventors:", len(co_inventor_ships))
    inventor_count = len(set([a for b in [x['patstat_ids'] for x in tmp] for a in b]))
    ai_inventor_count = len(set([a for b in [x['author_keys'] for x in tmp] for a in b]))
    print("Inventors:", inventor_count)
    print("AI-inventors:", ai_inventor_count)
    with open(output_path, 'w', encoding="utf-8") as f:
        for line in patent_dict.values():
            if len(line['author']) > 1:
                ujson.dump(un_set_lists(line), f)
                f.write("\n")


if __name__ == '__main__':
    from_inventions2patents()
    # load_patent_data(input_path="./data/large_data/German_AI_all2020-05-12.csv", output_dir="./data/pattrails/")



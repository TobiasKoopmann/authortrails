import torch
import ujson
from tqdm import tqdm
from util import from_paper2author
from transformers import AutoTokenizer, AutoModel

from collections import defaultdict


def get_scibert_paper_embeddings(years_remove: list = None,
                                 input_path: str = "./data/large_data/german_ai_community_dataset.json",
                                 output_path: str = "./data/large_data/scibert_paper_embeddings",
                                 relevant_keys: list = ('title', 'abstract', 'keywords')):
    def de_list_item(item):
        if isinstance(item, list) and len(item) == 1:
            return item[0]
        elif isinstance(item, list) and len(item) > 1:
            return " ".join(item)
        elif isinstance(item, list) and len(item) == 0:
            return ""
        return item

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
    len_dict = defaultdict(int)
    if years_remove[0] == "2015":
        output_path += "_5.csv"
    else:
        output_path += "_2.csv"
    with torch.no_grad():
        with open(input_path, 'r', encoding='utf-8') as in_f:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for line in tqdm(in_f):
                    line = ujson.loads(line)
                    if line['year'] not in years_remove:
                        key = line['key']
                        line = {key: de_list_item(line[key]) for key in relevant_keys if key in line}
                        line = " ".join(line.values())
                        bert_input = torch.tensor([tokenizer.encode(line, add_special_tokens=True, max_length=511)]).\
                            to(device)
                        len_dict[len(bert_input)] += 1
                        last_hidden_states = model(bert_input)[0]  # Models outputs are now tuples
                        last_hidden_states = [str(x) for x in last_hidden_states[-1][1].cpu().numpy()]
                        out_f.write(de_list_item(key) + ";" + ",".join(last_hidden_states) + "\n")
    print(len_dict)


y2015 = ["2015", "2016", "2017", "2018", "2019"]
y2018 = ["2018", "2019"]

if __name__ == '__main__':
    print("Getting Bert Paper embeddings. ")
    get_scibert_paper_embeddings(years_remove=y2015)
    get_scibert_paper_embeddings(years_remove=y2018)
    print("Getting Paper2Author Embeddings. ")
    from_paper2author(input_paper="./data/large_data/scibert_paper_embeddings_5.csv", filter_authors=True,
                      output_path="./data/german_author_scibert_embeddings_5.csv", years_remove=y2015)
    from_paper2author(input_paper="./data/large_data/scibert_paper_embeddings_2.csv", filter_authors=True,
                      output_path="./data/german_author_scibert_embeddings_2.csv", years_remove=y2018)
    print("#" * 100)
    print("Finished. ")

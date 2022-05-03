import os
import ujson
import math
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preview=True)
plt.rc('font', size=32)
plt.rc('legend', fontsize=32)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# plt.rc('axes', axisbelow=True)
params = {'text.latex.preamble': [r'\usepackage{amsfonts}']}
plt.rcParams.update(params)

ontology = {"baselines": ["true", "self", "random_coauthor"],
            "cognitive": ["lat_bert", "lat_svd", "lat_matrix", "lat_svd_singleDoc", "lat_matrix_singleDoc"],
            "institutional": ["institutional"],
            "organisational": ["affiliation", "url", "syntactic_web", "link_distance"],
            "social": ["diss_loc", "conference", "lat_deepwalk", "lat_node2vec_largeP", "lat_node2vec_smallP",
                       "lat_hope"],
            "geographic": ["geo_affiliation", "geo_city"]}

prox_colors = {"baselines": "dimgray",
               "cognitive": "cornflowerblue",
               "institutional": "magenta",
               "organisational": "red",
               "social": "mediumseagreen",
               "geographic": "gold"}

marker = itertools.cycle(('o', 'v', '^', '<', '>'))

color_alpha_marker = {}
for prox, methodlist in ontology.items():
    d = dict(zip(methodlist, np.linspace(1, 0.1, 8)))
    for method, alpha in d.items():
        color_alpha_marker[method.replace("_", " ")] = (prox_colors[prox], alpha, next(marker))

cut_off = 6


def read_one_evidence(path_to_directory: str) -> dict:
    """
    :param path_to_directory:
    :return:
    """
    if os.path.exists(path_to_directory + "evidence.json"):
        with open(path_to_directory + "evidence.json", "r", encoding="utf-8") as f:
            return ujson.load(f)
    else:
        raise FileNotFoundError


def svg2eps(filepath):
    pdf2psscript = "pdftops -eps " + filepath + ".pdf"
    os.system("bash -c '%s'" % pdf2psscript)


def scale_values(evidences: dict, baseline: str) -> dict:
    min_val, max_val = math.inf, 0
    for x in evidences.values():
        if max(x['evidence_values']) > max_val:
            max_val = max(x['evidence_values'])
        if min(x['evidence_values']) < min_val:
            min_val = min(x['evidence_values'])
    min_val = abs(min_val)
    max_val += min_val
    baseline_vals = evidences[baseline]['evidence_values']
    for k, x in evidences.items():
        tmp = x['evidence_values']
        x['evidence_values'] = [((float(x) + min_val + abs(baseline_vals[i])) / max_val) - 1 for i, x in enumerate(tmp)]
    return evidences


def rename_hypothesis(name: str, overview: bool = False, is_authortrails: bool = True) -> str:
    if overview:
        if is_authortrails:
            name = name.replace("url", "Org. Prox.")
            name = name.replace("lat svd", "Cog. Prox.")
            name = name.replace("lat deepwalk", "Soc. Prox.")
            name = name.replace("institutional", "Inst. Prox.")
            name = name.replace("geo city", "Geo. Prox.")
            name = name.title()
        else:
            name = name.replace("affiliation", "Org. Prox.")
            name = name.replace("lat svd", "Cog. Prox.")
            name = name.replace("lat hope", "Soc. Prox.")
            name = name.replace("institutional", "Inst. Prox.")
            name = name.replace("geo city", "Geo. Prox.")
            name = name.title()
    else:
        if "lat" in name:
            name = name.replace("lat", "")
        if "matrix" in name:
            if "single" in name:
                name = "C-NMF"
            else:
                name = "NMF"
        elif "svd" in name:
            if "single" in name:
                name = "C-LSA"
            else:
                name = "LSA"
        elif "link" in name:
            name = "Hyperlink"
        elif "syntactic" in name:
            name = "Hierarchy"
        elif "node" in name:
            if "small" in name:
                name = "Node2vec small p"
            else:
                name = "Node2vec large p"
        else:
            name = name.title()
    if "self" in name:
        name = "baseline"
    return name


def plot_hypothesis(plot_dict, output_path: str = None, hypothesis_not2plot: list = (), is_authortrails: bool = True,
                    image_name: str = "hyptrails.eps", overview: bool = False, scale: bool = False) -> None:
    """

    :param plot_dict:
    :param output_path:
    :param hypothesis_not2plot:
    :param is_authortrails:
    :param image_name:
    :param overview:
    :param scale:
    :return:
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    plot_values = {}
    title = "Dataset: " + image_name.replace("_", " ").replace(".eps", "").title()
    for name, values in plot_dict.items():
        if name == "max_id":
            title = title + " (total " + str(values) + " authors)"
        elif name == "k_values":
            # plt.xticks(np.arange(len(values)), np.array(values), rotation=0)
            pass
        if name not in hypothesis_not2plot:
            plot_values[name.replace("_", " ")] = values
    plots = []
    if scale:
        plot_values = scale_values(plot_values, baseline='random coauthor')
    for name, values in plot_values.items():
        color, alpha, marker = color_alpha_marker.get(name)
        if name != 'random coauthor':
            plots.append(
                plt.plot(np.array(plot_dict['k_values'])[0:cut_off], np.array(values['evidence_values'])[0:cut_off],
                         '-', linewidth=3, markersize=16, marker=marker, color=color, alpha=alpha,
                         label=rename_hypothesis(name, overview=overview, is_authortrails=is_authortrails)
                               # + "(" + str(values['data_points']) + ")"
                         )
                )
        elif not overview:
            plots.append(
                plt.plot(np.array(plot_dict['k_values'])[0:cut_off], np.array(values['evidence_values'])[0:cut_off],
                         'k-', linewidth=3, markersize=16, marker='.', color=color, alpha=alpha,
                         label=rename_hypothesis(name, overview=overview, is_authortrails=is_authortrails)
                               # + "(" + str(values['data_points']) + ")"
                         )
                )
        if is_authortrails:
            ax.set_ylim(bottom=0.0, top=0.5)
        else:
            ax.set_ylim(bottom=-0.15, top=0.5)
    print("Plotting ... ")
    ax.set_xlabel(r"Concentration factor $k$")
    ax.set_ylabel(r"Evidence")
    ax.legend(loc='upper center', handlelength=0.7, bbox_to_anchor=(0.5, 1.2),
              handletextpad=0.1, columnspacing=0.4, ncol=2, fancybox=True, shadow=True)
    ax.yaxis.grid()
    # ax.grid(True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('symlog', linthreshx=1, linscalex=0.6, subsx=(2, 3, 4, 5, 6, 7, 8, 9))
    # ax.set_xscale('log', basex=10)
    ax.set_xlim(left=0)
    # Either show or save, not both
    if output_path:
        if not os.path.exists(path=output_path):
            os.mkdir(path=output_path)
        print("Saving it to " + output_path + image_name)
        plt.savefig(output_path + image_name + ".pdf", dpi=300, bbox_inches='tight')
        plt.show()
        svg2eps(output_path + image_name)
    else:
        plt.show()
    plt.close()


SAVE = True
SCALE = True

if __name__ == '__main__':
    # for in_out in [["./data/authortrails_2/", "data/images/", "german_ai_dataset", True], ["./data/pattrails/", "data/images/", "patstat_dataset", False]]:
    for in_out in [["./data/pattrails/", "data/images/", "patstat_dataset", False]]:
        evidence = read_one_evidence(in_out[0])
        for part in ["organisational", "cognitive", "social"]:
            hypothesis_not2plot = ['k_values', 'max_id', 'true', 'conference', 'self']
            for onto, val in ontology.items():
                if onto != part and onto != "baselines":
                    hypothesis_not2plot.extend(val)
            if part == "cognitive":
                hypothesis_not2plot.extend(["lat_svd_singleDoc", "lat_matrix_singleDoc"])
            if SAVE:
                plot_hypothesis(copy.deepcopy(evidence), hypothesis_not2plot=hypothesis_not2plot, output_path=in_out[1],
                                image_name=in_out[2] + "_" + part, scale=SCALE, is_authortrails=in_out[3])
            else:
                plot_hypothesis(copy.deepcopy(evidence), hypothesis_not2plot=hypothesis_not2plot,
                                image_name=in_out[2] + "_" + part, scale=SCALE, is_authortrails=in_out[3])
        hypothesis_not2plot = ['k_values', 'max_id', 'self']
        hypothesis_not2plot.extend(["lat_bert", "lat_matrix", "lat_svd_singleDoc", "lat_matrix_singleDoc"])  # rm cog
        if in_out[3]:
            hypothesis_not2plot.extend(["conference", "affiliation", "syntactic_web", "link_distance", "diss_loc"])  # rm orga
            hypothesis_not2plot.extend(["lat_hope", "lat_node2vec_largeP", "lat_node2vec_smallP"])  # social
        else:
            hypothesis_not2plot.extend(["conference", "url", "syntactic_web", "link_distance", "diss_loc"])  # rm orga
            hypothesis_not2plot.extend(["lat_deepwalk", "lat_node2vec_largeP", "lat_node2vec_smallP"])  # social
        hypothesis_not2plot.extend(["geo_affiliation"])  # too few data points
        if SAVE:
            plot_hypothesis(copy.deepcopy(evidence), hypothesis_not2plot=hypothesis_not2plot, output_path=in_out[1],
                            image_name=in_out[2], overview=True, scale=SCALE, is_authortrails=in_out[3])
        else:
            plot_hypothesis(copy.deepcopy(evidence), hypothesis_not2plot=hypothesis_not2plot, image_name=in_out[2],
                            overview=True, scale=SCALE, is_authortrails=in_out[3])

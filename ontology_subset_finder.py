from aggregator import CorrectRatio
from copy import deepcopy
import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

def table_to_graph(table):
    G = nx.Graph()
    col_id_offset = len(table["table_names"])
    G.add_nodes_from(range(col_id_offset + len(table["column_names"])))
    for col_num, (par_num, col_name) in enumerate(table["column_names"]):
        if par_num == -1:
            continue
        G.add_edge(col_num + col_id_offset, par_num, weight=1)
    for f, p in table["foreign_keys"]:
        G.add_edge(f + col_id_offset, p + col_id_offset, weight=1)
    G.add_node(-1)
    for table_num in range(len(table["table_names"])):
        G.add_edge(-1, table_num, weight=99999)
    G.add_edge(-1, col_id_offset, weight=99999)
    return G


def find_ontology_subset_all(data, tables, log=True):
    top_match_ratio = CorrectRatio()
    candidate_match_ratio = CorrectRatio()
    for db_id in tables:
        tables[db_id]["graph"] = table_to_graph(tables[db_id])
    for datum_idx, datum in enumerate(tqdm(data)):
        table = tables[datum["db_id"]]
        top_match, candidate_match = find_ontology_subset(datum, table, log)
        top_match_ratio.update(top_match)
        candidate_match_ratio.update(candidate_match)

    return top_match_ratio.get_ratio(), candidate_match_ratio.get_ratio()


def table_printer(table):
    for tab_num, tab_name in enumerate(table["table_names"]):
        print("TABLE: {}".format(tab_name))
        for col_num, (par_tab, col_name) in enumerate(table["column_names"]):
            if tab_num == par_tab:
                print("  {}: {}".format(col_num, col_name))
    print("constraint: {}".format(table["foreign_keys"]))


def find_ontology_subset(datum, table, log=True):
    question_toks = datum["question_toks"]
    evidence_nodes = find_evidence_nodes(question_toks, table)
    spanned_tree = []
    gold_ontology_subset = get_gold_ontology_subset(datum["sql"])
    candidate_correct = False
    for candidate_set in evidence_nodes:
        col_offset = len(table["table_names"])
        table_nodes = list(candidate_set["tables"])
        col_nodes = [col_num + col_offset for col_num in candidate_set["cols"]]
        tree = steiner_tree(table["graph"], table_nodes + col_nodes)
        nodes = tree.nodes()
        tables = set()
        cols = set()
        if not nodes:
            tables = set(table_nodes)
            cols = set(col_nodes)
        for node in nodes:
            if node < col_offset:
                tables.add(node)
            else:
                cols.add(node - col_offset)
        spanned_tree.append({'tables': tables, 'cols': cols})
        if gold_ontology_subset["tables"] == tables and gold_ontology_subset["cols"] == cols:
            candidate_correct = True
    if log:
        print("======================")
        table_printer(table)
        print(datum["question"])
        print(datum["query"])
        print(gold_ontology_subset)
        print(evidence_nodes)
        print(spanned_tree)
        print(candidate_correct)
    return True, candidate_correct


def find_matching_nodes(word_toks, table):
    matching_nodes = []
    n = WordNetLemmatizer()
    word = ' '.join(word_toks)
    lemmatized_word = [n.lemmatize(w) for w in word_toks]
    lemmatized_word = ' '.join(lemmatized_word)
    table_names = table["table_names"]
    col_names = [tab_col[1] for tab_col in table["column_names"]]
    for table_num, table_name in enumerate(table_names):
        lemmatized_table_name = table_name.split(' ')
        lemmatized_table_name = [n.lemmatize(w) for w in lemmatized_table_name]
        lemmatized_table_name = ' '.join(lemmatized_table_name)
        if word == table_name:
            matching_nodes.append({
                "tables": {table_num},
                "cols": set()
            })

        elif lemmatized_word == lemmatized_table_name:
            matching_nodes.append({
                "tables": {table_num},
                "cols": set()
            })
    for col_num, col_name in enumerate(col_names):
        lemmatized_col_name = col_name.split(' ')
        lemmatized_col_name = [n.lemmatize(w) for w in lemmatized_col_name]
        lemmatized_col_name = ' '.join(lemmatized_col_name)

        if word == col_name:
            matching_nodes.append({
                "tables": set(),
                "cols": {col_num}
            })
        elif lemmatized_word == lemmatized_col_name:
            matching_nodes.append({
                "tables": set(),
                "cols": {col_num}
            })

        # elif lemmatized_word in lemmatized_col_name:
        #     matching_nodes.append({
        #         "tables": set(),
        #         "cols": {col_num}
        #     })
        #     matching_nodes.append({
        #         "tables": set(),
        #         "cols": set()
        #     })

    return matching_nodes


def find_evidence_nodes(question_toks, table):
    def check_exist_and_add(node_list, node):
        for existing_node in node_list:
            if existing_node['tables'] == node['tables'] and existing_node['cols'] == node['cols']:
                return
        node_list.append(node)

    if not question_toks:
        return []
    candidate_nodes_list = []
    max_n_gram = min(6, len(question_toks) - 1)
    one_matched = False
    for n_gram in range(max_n_gram, 0, -1):
        word = question_toks[:n_gram]
        matching_nodes = find_matching_nodes(word, table)
        if matching_nodes:
            remaining_evidence_nodes = find_evidence_nodes(question_toks[n_gram:], table)
            if not remaining_evidence_nodes:
                for node in matching_nodes:
                    check_exist_and_add(candidate_nodes_list, node)
            else:
                for node in matching_nodes:
                    for candidate_set in remaining_evidence_nodes:
                        newset = {
                            "tables": candidate_set["tables"] | node["tables"],
                            "cols": candidate_set["cols"] | node["cols"]
                        }
                        check_exist_and_add(candidate_nodes_list, newset)
            one_matched = True
            break
    if not one_matched:
        new_nodes = find_evidence_nodes(question_toks[1:], table)
        for node in new_nodes:
            check_exist_and_add(candidate_nodes_list, node)
    return candidate_nodes_list


def get_gold_ontology_subset(sql):
    def find_all_table_nums(sql):
        if isinstance(sql, list):
            for item in sql:
                yield from find_all_table_nums(item)
        if isinstance(sql, dict):
            for keyword in sql:
                if keyword == "table_units":
                    for _, table_num in sql["table_units"]:
                        if not isinstance(table_num, int):
                            yield from find_all_table_nums(table_num)
                        else:
                            yield table_num
                else:
                    yield from find_all_table_nums(sql[keyword])

    def find_all_col_nums(sql):
        def yield_for_col_unit(col_unit):
            if not col_unit:
                return False
            agg_id, col_id, isDistinct = col_unit
            yield col_id

        def yield_for_val_unit(val_unit):
            unit_op, col_unit1, col_unit2 = val_unit
            yield from yield_for_col_unit(col_unit1)
            yield from yield_for_col_unit(col_unit2)

        if not sql:
            return
        for cond in sql["from"]["conds"]:
            if isinstance(cond, str):
                continue
            not_op, op_id, val_unit, val1, val2 = cond
            yield from yield_for_val_unit(val_unit)
            agg_id, col_id, isDistinct = val1
            yield col_id

        for _, table_num in sql["from"]["table_units"]:
            if isinstance(table_num, dict):
                yield from find_all_col_nums(table_num)
                continue
        for val_unit in sql["select"][1]:
            agg_id, val_unit = val_unit
            yield from yield_for_val_unit(val_unit)
        for cond in sql["where"]:
            if isinstance(cond, str):
                continue
            not_op, op_id, val_unit, val1, val2 = cond
            yield from yield_for_val_unit(val_unit)
            if isinstance(val1, dict):
                yield from find_all_col_nums(val1)
        for col_unit in sql["groupBy"]:
            yield from yield_for_col_unit(col_unit)
        for cond in sql["having"]:
            if isinstance(cond, str):
                continue
            not_op, op_id, val_unit, val1, val2 = cond
            yield from yield_for_val_unit(val_unit)
            if isinstance(val1, dict):
                yield from find_all_col_nums(val1)
            if isinstance(val1, dict):
                yield from find_all_col_nums(val1)
        if sql["orderBy"]:
            for val_unit in sql["orderBy"][1]:
                yield from yield_for_val_unit(val_unit)

        yield from find_all_col_nums(sql["except"])
        yield from find_all_col_nums(sql["intersect"])
        yield from find_all_col_nums(sql["union"])

    return {
        "tables": set(find_all_table_nums(sql)),
        "cols": set(find_all_col_nums(sql)) - {0}
    }

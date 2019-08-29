import json
from ontology_subset_finder import find_ontology_subset_all


def load_tables(table_path):
    data = json.load(open(table_path))
    table = dict()
    for item in data:
        table[item["db_id"]] = item
    return table


if __name__ == "__main__":
    tables = load_tables("./data/tables.json")
    train_data = json.load(open("./data/train_all.json"))
    dev_data = json.load(open("./data/dev.json"))

    # train_top_matching, train_candidate_matching = find_ontology_subset_all(train_data, tables, log=True)
    dev_top_matching, dev_candidate_matching = find_ontology_subset_all(dev_data, tables, log=True)

    # print("Training - top: {} / candidate: {}".format(train_top_matching, train_candidate_matching))
    print("Dev - top: {} / candidate: {}".format(dev_top_matching, dev_candidate_matching))

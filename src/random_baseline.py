# import pandas as pd
# import json

def random_baseline(json_data, key):
    length_of_doc = len(json_data[key])
    number_of_page = sum(json_data[key])
    if not number_of_page // length_of_doc:
        return [number_of_page // length_of_doc for _ in range(length_of_doc)]
    else:
        return [number_of_page // length_of_doc for _ in range(length_of_doc-1)] + [number_of_page % length_of_doc + number_of_page // length_of_doc]



# with open('../corpus1/TrainTestSet/Trainset/Doclengths_of_the_individual_docs_TRAIN.json', 'rb') as f:
#     d1 = json.load(f)

# with open('../corpus2/TrainTestSet/Trainset/Doclengths_of_the_individual_docs_TRAIN.json', 'rb') as f:
#     d2 = json.load(f)


# d3 = dict(d1)
# d3.update(d2)

# print(len(d1), len(d2), len(d3))

# print(sum(random_baseline(d3, key='893872')))
# print(sum(d3['893872']))
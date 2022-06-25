import numpy as np


def f1_calculator(cf_matrix, cls=1, os=1e-6):
    tp = cf_matrix[cls, cls]
    fp = cf_matrix[1-cls, cls]
    tn = cf_matrix[1-cls, 1-cls]
    fn = cf_matrix[cls, 1-cls]
    precision = tp/(tp+fp+os)
    recall = tp/(tp+fn+os)
    f1_score = 2*(precision*recall)/(precision+recall)
    return f1_score

def evaluation(cf_matrix, cls=1):
    tp = cf_matrix[cls, cls]
    fp = cf_matrix[1-cls, cls]
    tn = cf_matrix[1-cls, 1-cls]
    fn = cf_matrix[cls, 1-cls]
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(precision*recall)/(precision+recall)
    print("F1: ", f1_score)
    sen = tp/(tp+fn)
    print("Sen: ", sen)
    spec = tn/(tn+fp)
    print("Spec: ", spec)
    acc = (tp+tn)/(tp+tn+fp+fn)
    print("Acc: ", acc)

a = np.array([[2165, 66], [86, 356]])
evaluation(a)
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from fastDamerauLevenshtein import damerauLevenshtein
from sklearn.metrics import f1_score, precision_score, recall_score

# functie om van lijst van lengtes naar binair te gaan
def length_list_to_bin(list_of_lengths: list) -> np.array:
    out = np.zeros(shape=(sum(list_of_lengths)))
    out[0] = 1
    # last document boundary is assumed
    if len(list_of_lengths) == 1:
        return out
    out[np.cumsum(list_of_lengths[:-1])] = 1
    return out


def bin_to_length_list(binary_vector: list) -> np.array:
    # We basically reverse the operation
    bounds = binary_vector.nonzero()[0]
    if not len(bounds):
        return np.array([len(binary_vector)])
    # fix final element
    bounds = np.append(bounds, len(binary_vector))
    
    # get consecutive indices
    return np.ediff1d(bounds)


def make_index(binary_vec: np.array):
    # make index variant for binary vectors. we First get a split of elements by
    # using np.split and using the indices of the boundary pages.
    splits = np.split(np.arange(len(binary_vec)), binary_vec.nonzero()[0][1:])
    repeated_splits = np.repeat(np.array(splits, dtype=object), [len(split) for split in splits], axis=0)
    # Now we have a list of splits, we repeat this split n times, where n is the length of the split
    out = { i: set(item) for i, item in enumerate(repeated_splits)}
    
    return out


def Bcubed(truth, pred, aggfunc=np.mean):
    assert len(truth)==len(pred)  # same amount of pages
    truth,pred = make_index(truth), make_index(pred)
    
    df  ={i:{'size':len(truth[i]),'P':0,'R':0,'F1':0} for i in truth}
    for i in truth:
        df[i]['P']= len(truth[i] & pred[i])/len(pred[i]) 
        df[i]['R']= len(truth[i] & pred[i])/len(truth[i])
        df[i]['F1']= (2*df[i]['P']*df[i]['R'])/(df[i]['P']+df[i]['R'])
     
    return  pd.DataFrame(df).T


def HammingDamerau(gold, prediction):
    assert len(gold) == len(prediction)
    return damerauLevenshtein("".join([str(item) for item in gold]), "".join([str(item) for item in prediction]),
                              similarity=False, insertWeight=10**5, deleteWeight=10**5) / len(gold)

from numpy.lib.stride_tricks import sliding_window_view
def window_diff(gold: np.array, prediction: np.array) -> float:
    
    assert len(gold) == len(prediction)
    # laten we in dit geval ervanuit gaan dat we k berekenen per document
    # En niet over het hele corpus.
    
    
    k = int(bin_to_length_list(gold).mean()*1.5)
    # small check, in case of a singleton cluster, k will be too large
    # (mean == doc_length, k = 1.5*doclength)
    if k > len(gold):
        k = len(gold)
    
    # met de numpy functie kunnen we sliding windows pakken
    # dit doen we voor allebei de arrays en die vergelijken we dan.
    
    gold_windows = sliding_window_view(gold, window_shape=k)
    pred_windows = sliding_window_view(prediction, window_shape=k)
    
    # nu moeten we dus per window kijken of voor beiden de som gelijk is.
    gold_sum = gold_windows.sum(axis=1)
    pred_sum = pred_windows.sum(axis=1)
    
    # nu hebben we de som voor elke window in allebei de arrays
    # de score is nu gelijk aan de mean van de bool array
    
    return (gold_sum != pred_sum).mean()



def block_precision(gold: np.array, prediction: np.array) -> float:
    
    # hier gebruiken we np split. we splitten een stream van 1,2, 3, .., n
    # op aan de hand van de indices  nonzeros in de binaire vectors.
    # vervolgens maken we van deze partities 2 sets met subsets en 
    # berekenen de grootte van de intersectie
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)), prediction.nonzero()[0][1:])
    
    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])
    
    return len(gold_set & pred_set) / len(pred_set)


def block_recall(gold: np.array, prediction: np.array) -> float:
    
    # hier gebruiken we np split. we splitten een stream van 1,2, 3, .., n
    # op aan de hand van de indices  nonzeros in de binaire vectors.
    # vervolgens maken we van deze partities 2 sets met subsets en 
    # berekenen de grootte van de intersectie
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)), prediction.nonzero()[0][1:])
    
    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])
    
    return len(gold_set & pred_set) / len(gold_set)

def block_F1(gold: np.array, prediction: np.array) -> float:
    
    # hier gebruiken we np split. we splitten een stream van 1,2, 3, .., n
    # op aan de hand van de indices  nonzeros in de binaire vectors.
    # vervolgens maken we van deze partities 2 sets met subsets en 
    # berekenen de grootte van de intersectie
    gold_splits = np.split(np.arange(len(gold)), gold.nonzero()[0][1:])
    pred_splits = np.split(np.arange(len(prediction)), prediction.nonzero()[0][1:])
    
    # set of sets dan kunnen we makkelijk kijken welke subsets precies overeen komen
    gold_set = set([frozenset(item) for item in gold_splits])
    pred_set = set([frozenset(item) for item in pred_splits])
    
    P = len(gold_set & pred_set) / len(pred_set)
    R = len(gold_set & pred_set) / len(gold_set)
    
    # We have to be careful here, if both Precision and recall are zero
    # we just return 0
    if P == 0 and R == 0:
        return 0
    
    return (2*P*R) / (P+R)

def f1(gold: np.array, prediction: np.array) -> float:
    return f1_score(gold, prediction)

def precision(gold: np.array, prediction: np.array) -> float:
    return precision_score(gold, prediction)

def recall(gold: np.array, prediction: np.array) -> float:
    return recall_score(gold, prediction)



def make_index_doc_lengths(split):
    l= sum(split)
    pages= list(np.arange(l))
    out = defaultdict(set)
    for block_length in split:
        block= pages[:block_length]
        pages= pages[block_length:]
        for page in block:
            out[page]= set(block)
    return out

def IoU_TruePositives(t,h):
    '''A True Positive is a pair h_block, t_block with an IoU>.5.
    This function returns the sum of all IoUs(h_block,t_block) for these bvlocks in t and h.'''
    def IoU(S,T):
        '''Jaccard similarity between sets S and T'''
        return len(S&T)/len(S|T)
    def get_docs(t):
        '''Get the set of documents (where each document is a set of pagenumbers)'''
        return {frozenset(S) for S in make_index_doc_lengths(t).values()}
    def find_match(S,Candidates):
        '''Finds, if it exists,  the unique T in Candidates such that IoU(S,T) >.5'''
        return [T for T in Candidates if IoU(S,T) >.5]
    t,h= get_docs(t), get_docs(h) # switch to set of docs representation
    return sum(IoU(S,find_match(S,t)[0]) for S in h if find_match(S,t))


def IoU_P(t,h):
    return IoU_TruePositives(t,h)/len(h)


def IoU_R(t,h):
    return IoU_TruePositives(t,h)/len(t)


def IoU_F1(t,h):
    P,R= IoU_P(t,h),IoU_R(t,h)
    #todo, add the direct definition using FPs and FNs as well.
    # and test they are indeed equal
    return 0 if (P+R)==0 else 2*P*R/(P+R)



def calculate_metrics_one_stream(gold_vec, prediction_vec):
    
    out = {}
    
    gold_vec = np.array(gold_vec)
    prediction_vec = np.array(prediction_vec)
    scores = {'Boundary': f1(gold_vec, prediction_vec),
             'Bcubed': Bcubed(gold_vec, prediction_vec)['F1'].mean(),
             'WindowDiff': 1-window_diff(gold_vec, prediction_vec),
             'Block': block_F1(gold_vec, prediction_vec),
             'Weighted Block': IoU_F1(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec))}
    
    scores_precision = {'Boundary': precision(gold_vec, prediction_vec),
             'Bcubed': Bcubed(gold_vec, prediction_vec)['P'].mean(),
             'WindowDiff': 1-window_diff(gold_vec, prediction_vec),
             'Block': block_precision(gold_vec, prediction_vec),
             'Weighted Block': IoU_P(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec))}

    scores_recall = {'Boundary': recall(gold_vec, prediction_vec),
             'Bcubed': Bcubed(gold_vec, prediction_vec)['R'].mean(),
             'WindowDiff': 1-window_diff(gold_vec, prediction_vec),
             'Block': block_recall(gold_vec, prediction_vec),
             'Weighted Block': IoU_R(bin_to_length_list(gold_vec), bin_to_length_list(prediction_vec))}
        
    out['precision'] = scores_precision
    out['recall'] = scores_recall
    out['F1'] = scores
        
    return out


def calculate_scores_df(gold_standard_dict, prediction_dict):
    all_scores = defaultdict(dict)
    for key in gold_standard_dict.keys():
        metric_scores = calculate_metrics_one_stream(gold_standard_dict[key], prediction_dict[key])
        for key_m in metric_scores.keys():
            all_scores[key_m][key] = metric_scores[key_m]  
    return {key: pd.DataFrame(val) for key, val in all_scores.items()}


def calculate_mean_scores(gold_standard_dict, prediction_dict, show_confidence_bounds=True):
    
    scores_df = {key: val.T.mean().round(2) for key, val in calculate_scores_df(gold_standard_dict, prediction_dict).items()}
    scores_combined = pd.DataFrame(scores_df)
    test_scores = scores_combined
    
    confidence = 0.95
    
    # total number of documents is the number of ones in the binary array
    n = sum([np.sum(item) for item in prediction_dict.values()])
    
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((test_scores * (1 - test_scores)) / n )

    ci_lower = (test_scores - ci_length).round(2)
    ci_upper = (test_scores + ci_length).round(2)
    
    precision_ci = ci_lower['precision'].astype(str) + '-' + ci_upper['precision'].astype(str)
    recall_ci = ci_lower['recall'].astype(str) + '-' + ci_upper['recall'].astype(str)
    f1_ci = ci_lower['F1'].astype(str) + '-' + ci_upper['F1'].astype(str)
    
    out = pd.DataFrame(scores_df)
    out = out.rename({0: 'value'}, axis=1)
    out['support'] = sum([np.sum(item) for item in gold_standard_dict.values()])
    if show_confidence_bounds:
        out['CI Precision'] = precision_ci
        out['CI Recall'] = recall_ci
        out['CI F1'] = f1_ci
    

    return out

def show_KDE_plots(gold_standard_dict, prediction_dict, save_name=""):
    # Make figures and axes here and plot
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    
    scores = calculate_scores_df(gold_standard_dict, prediction_dict)
    
    for i, (key, val) in enumerate(scores.items()):

        val.T.plot(kind='kde', title="%s KDE for metrics" % key, ax=axes[i],
                  xlim=[0.0, 1.0])
    if save_name:
        plt.savefig(save_name)
    plt.show()


    
def evaluation_report(gold_standard_json, prediction_json, round_num=2,
                     title="", show_confidence_bounds=True):
    # 1. print the mean scores
    display(Markdown("<b> Mean scores of the evaluation metrics for %s </b>" % title))
    display(calculate_mean_scores(gold_standard_json, prediction_json,
                                 show_confidence_bounds=show_confidence_bounds).round(round_num))

    display(Markdown("<b> KDE Plots of the scores of the evaluation metrics for %s </b>" % title))
    # 2. Plot the KDEs
    show_KDE_plots(gold_standard_json, prediction_json)

    
def _convert_to_start_middle_end(binary_stream):
    out = []
    length_list = bin_to_length_list(np.array(binary_stream))
    for doc in length_list:
        if doc > 1:
            out.extend([1]+ [0]*(doc-2)+[2])
        else:
            out.extend([1])
    assert len(binary_stream) == len(out)
    return np.array(out)

def show_confusion_matrix(gold_standard_json, prediction_json):
    translation_dict = {1: 'start page', 0: 'middle page', 2: 'last page'}
    gold = []
    prediction = []
    for key in gold_standard_json.keys():
        bin_gold = [translation_dict[item] for item in _convert_to_start_middle_end(gold_standard_json[key].tolist())]
        bin_pred = [translation_dict[item] for item in prediction_json[key].tolist()]
        
        gold.extend(bin_gold)
        prediction.extend(bin_pred)
    
    mat = ConfusionMatrixDisplay.from_predictions(prediction, gold)
    mat.ax_.set_ylabel('Prediction')
    mat.ax_.set_xlabel('Ground Truth')
    plt.show()
# README Document Splitting Challenge 
Document Splitting

Data Link : https://www.icloud.com/iclouddrive/081jNFZK6tVBJYKq8V3D1J2rw#Data

Final Thesis report: https://www.overleaf.com/5948359792rndwxymmpbjx


### Content of folder

<pre>
bash-3.2$ cd DocumentSplittingChallenge/
bash-3.2$ du -hs *
8.5G    corpus1
7.3G    corpus2
bash-3.2$ ls */*/*
corpus1/TrainTestSet/Testset.zip        corpus2/TrainTestSet/Testset.zip

corpus1/TrainTestSet/Trainset:
Doclengths_of_the_individual_docs_TRAIN.json    data
VerzoekEnBesluit                                ocred_text.csv.gz

corpus2/TrainTestSet/Trainset:
Doclengths_of_the_individual_docs_TRAIN.json    ocred_text.csv.gz
data
</pre>

The folder contains 2 traintest sets, corpus1 and corpus2. Both have the same setup:

* a testset that you cannot see, but which consists of similar files as the trainset (also from the same source)
* a trainset that you can see, containing
    * gold standard in `json`
       * for every filename a list of the lengths of the individual docs (in order). You cabn check that the sum of the lengths in the list for pdf doc A is equal to the number of pages in A.
   * the data: a set of pdfs, whose filename is (almost) equal to the keys in the gold standard json
   * additional information (eg in VerzoekEnBesluit) on each wob-request.
   * `ocred_text.csv.gz` the ocred text from each page in each pdf in the corpus 


## Challenge

* Create software that can split such PDFs. The output should be a json file with the same structure as the gold standard.
* Your software needs to run as is on a computer provided by the organisation of the challenge, on a folder with the same structure as the Trainset. 
    * A sandbox will be provided in which you can try out your code.


## Evaluation

* The performance of your splitter is measured by mean Bcubed F1 and mean Hamming distance over all docs in the two testsets. So each run obtains four evaluation measures.

## How to run model
### For training
Run python script mlp_classification_train.py with variable parser arguments:
```
--project [project name]: where the model and logging are stored, default location ./models/mlp_model/[project name]
--model [model name]: choosing model, vgg_lstm (VGG-LSTM) or bert (VGG-BERT)
--weighted [weighted dir]: load pretrained weighted, if you don't want to use pretrained weighted, ignore this argument
--device [select device]: cuda or cpu
--epochs [num epochs]: setting number of training epochs
--batch_size [num batch]: setting batch size
--l_r [learning rate]: setting learning rate
--loss [loss function]: focal or bce
--optim [optimizer]: sam, adam or sgd
--augment [bool]: True or False, augment training data, only class 1
```

For example, train data with VGG-BERT model, without pretrained weighted, running on GPU, set epochs is 350, batch size 64, learning rate 0.001, focal loss, sam optimizer and augment training data, saving location ./models/mlp_model/train_bert: 

```
python mpl_classification_train.py --model bert --device cuda --epochs 350 --batch_size 64 --l_r 0.001 --loss focal --optim sam --augment True --project train_bert
```
to change to VGG-LSTM, replace 
```
--model bert
```
by 
```
--model vgg_lstm
```

to change to VGG only, replace 
```
--model bert
```
by 
```
--model vgg_only
```

to change to LSTM only, replace 
```
--model bert
```
by 
```
--model lstm_only
```

to change to BERT only, replace 
```
--model bert
```
by 
```
--model bert_only
```

### For testing
Run python script mlp_classification_train.py with variable parser arguments:
```
--project [project name]: where the confusion matrix are stored, default location ./models/mlp_model/[project name]
--model [model name]: choosing model, vgg_lstm (VGG-LSTM) or bert (VGG-BERT)
--weighted [weighted dir]: load pretrained weighted
--device [select device]: cuda or cpu
--batch_size [num batch]: test batch size
--mode [mode run]: test or train, default is train
```

For example, test VGG-BERT model with best vailidation loss model, running on CPU, test batch size 256, 
weighted location 
```
./models/mlp_model/train_bert/best_val_loss.pt
```

saving location 
```
./models/mlp_model/test_bert
```

```
python mlp_classificaiton_train.py --mode test --project test_bert --device cpu --batch_size 256 --weighted ./models/mlp_model/train_test/best_val_loss.pt
```


## References
- [https://arxiv.org/pdf/1910.03678.pdf](https://arxiv.org/pdf/1910.03678.pdf)
- [https://link.springer.com/chapter/10.1007/978-3-540-76280-5_5](https://link.springer.com/chapter/10.1007/978-3-540-76280-5_5)
- [https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
- [https://proceedings.neurips.cc/paper/2021/file/0084ae4bc24c0795d1e6a4f58444d39b-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/0084ae4bc24c0795d1e6a4f58444d39b-Paper.pdf)

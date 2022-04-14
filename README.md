# README Document Splitting Challenge 
Document Splitting

Data Link : https://www.icloud.com/iclouddrive/081jNFZK6tVBJYKq8V3D1J2rw#Data

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

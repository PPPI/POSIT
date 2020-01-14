# POSIT
This a project to simultaneously provide language ID tags and Part-Of-Speech 
or compiler tags (which are taken from CLANG compilations of C and C++ code).

The corpus is either code with comments annotated with CLANG compiler information 
and universal PoS tags for English, or StackOverflow. For StackOverflow we start from
the data dump (which can be found [here](https://archive.org/details/stackexchange)), 
and use a frequency based heuristic to annotate code snippets. The frequency data
is made available under `./data/corpora/SO/frequency_map.json`. To generate training
data from StackOverflow, please use the scripts under `./src/preprocessor` together 
with the `Posts.xml` file from the data dump linked above.

Our model is a BiLSTM neural network with a Linear CRF and Viterbi decode to go from
LSTM state to tags or language IDs. We use the same LSTM network and change only the CRF
on top for the two tasks. We linearly combine the two objectives in the loss with a 
slightly smaller weight given to language IDs. We do not condition Tag output on 
language IDs in this version of the model.

## Dependencies
This project requires the following python libraries:
```
tensorflow<=1.15, numpy, gensim, nltk
```
for the main model (mind that tensorflow 1 is used within this repository);
```
xmltodict, beautifulsoup4, html5lib
```
for corpus preprocessing; and
```
scikit-learn
```
for the considered SVM baseline.

## Repository Structure
The repo is structured as follows:
```
src/ -> baseline/        {rulebased and SVM baselines}
        -> StORMeD/      {StORMeD[1] client and adapted baselines}     
        preprocessor/    {corpus construction scripts for the two corpora used}
        tagger/          {Main model code and utils}
        evaluate.py      {Script to evaluate the main model for current config}
        export_Model.py  {Script to export a TensorFlow model (for reuse in other TF systems)}
        process_pairs.py {Script to process (input, oracle) pairs}
        process.py       {Script to process posts into a text file for further use}
        RPC_serve.py     {Script to run a server accesible via RPC for reuse in other systems}
        train.py         {Script to train a model given current config}
```
The model configuration and hyperparameters can be seen in `./src/tagger/config.py`

## Preprocessing
To generate training data from `Posts.xml`, first run:
```bash
$ python src/preprocessor/preprocess.py <path of Posts.xml> <output name> <start> <end> <frequency?> <language id?>
```
The second argument should be one of `train/dev/eval` to indicate what part of the data
it represents. The third and fourth arguments represent the offset indices into the 
`Posts.xml` file; from which post to which post. The last two arguments are either `true` 
or `false` and indicate if (1) the `frequency_map.json` file should be used in tagging 
and (2) if language IDs should be recorded in the output files.

After generating the `train/dev/eval.txt` files, one should also generate the dictionary
files needed to convert to and from integers. This can be done as such:
```bash
$ python src/preprocessor/generaty_vocabulary.py <corpus name> <language id?>
```

For the CodeComment corpus, please use the zip: `data/corpora/lucid.zip`. 
To generate the training data, please use the lucid scripts under `src/preprocessor`.
For example:
```bash
     Convert from .lucid json to .txt ------v
[../posit] $ python src/preprocessor/lucid_reader.py ./data/corpora/lucid false
                                     corpus location (unzipped) --^           ^--- Use Language IDs?
                                            v--------- Consolidate the txt file into train, dev and eval
[../posit] $ python src/preprocessor/lucid_preprocessor.py ./data/corpora/lucid false
                                           corpus location (unzipped) --^           ^--- Use Language IDs?
               Generate vocabulary files ------v
[../posit] $ python src/preprocessor/generate_vocabulary.py lucid false
                            corpus name (under data/corpora) --^      ^--- Use Language IDs?
```

## Running the BiLSTM model
To run the model, make sure that the necessary `train/dev/eval.txt` files have been generated with 
the pre-processor scripts and update `./src/tagger/config.py` to point to them. Once that is done, simply
run:

```bash
$ python src/train.py
```
Similarly `evaluate.py` can be run after a training session, but then as a cli argument the location of the 
model.weights folder needs to be provided:
```bash
$ python src/evaluate.py ./<path to model.weights>/<uuid>/model.weights>
```
The path to model.weights is usually under results/<corpus Name>

Similarly to `evaluate.py`, `process.py` can be run; however, it requires a further argument:
```bash
$ python src/process.py ./<path to model.weights> ./<path to data to process>
```

## Running the Rule-based and SVM Baselines
For the baselines, please run as follows:
```bash
$ python src/baseline/classification.py <corpus name> <use SVMs?>
```
The second argument should be either `true` or `false`.


## Running the StORMeD baselines
For the StORMeD comparisons, there are multiple scripts depending on the scenario.

For the performance on the evaluation set, run:
```bash
$ python src/baseline/StORMeD/stormed_query_from_eval.py <corpus name> <StORMeD api key>
```

To query with posts from the StackOverflow Data-dump directly, run:
 ```bash
$ python src/baseline/StORMeD/stormed_query_so.py <path to Posts.xml> <offset into Posts.xml> <StORMeD api key>
```

To query stormed on manually selected posts where a human oracle is available, run:
 ```bash
$ python src/baseline/StORMeD/stormed_query_local_so.py <path to selected_Ids_and_revisions.csv> <StORMeD api key>
```

Mind that in all scenarios a API key must be used. As these are associated to an e-mail, we cannot provide one; however,
they can be requested from the official [StORMeD website](https://stormed.inf.usi.ch/). 

The use of the StORMeD API is not restricted only to this project and the interested person should consider other 
use-cases as well. The documentation provided Ponzanelli and others is extensive and this project serves as a demo of 
using it from within Python.

The client code is a slightly adapted version of the python client provided by Ponzanelli et al.

## References

[1] Ponzanelli, L., Mocci, A., & Lanza, M. (2015). 
[StORMeD: Stack overflow ready made data.](https://stormed.inf.usi.ch/) 
IEEE International Working Conference on Mining Software Repositories, 
2015-Augus, 474–477. https://doi.org/10.1109/MSR.2015.67
# Question Answering on SQuAD dataset - Stanford CS224n Win 2018

## Description
Code for the Default Final Project (SQuAD) for [CS224n](http://web.stanford.edu/class/cs224n/), Winter 2018. The goal of the project is to tackle the question answering problem on the Stanford Question Answering Dataset. 
Given a short text (context) and a reading comprehension question, the model is expected to output the answer to the question. As a simplification, in the SQuAD dataset, the answer is a subtext of the context. 

Details on the implementation, experiments and models can be found in the [report](http://web.stanford.edu/class/cs224n/reports/6904508.pdf) and the [poster](https://drive.google.com/file/d/1zW3rdprpoyvh9kB04Jgw3lpkV0sNjNlR/view?usp=sharing).

## Code description 
The code was developed and tested with Python 2.7 and Tensorflow 1.4.

### Setup 

```
./get_started.sh
```
This creates a conda environment called `squad`, downloads the dataset and word embeddings and setups the requirements. 

### Models 
For this project we tested the following model architectures 
* Baseline: The provided baseline is composed of:
- Embedding layer: The question and context inputs are processed using word embeddings vectors (GloVe)
- Encoding layer: A bidirectional RNN layer is used to encode both question and context.
- Attention layer: A context to question attention distribution is computed.
- Modeling layer: The attention output and the context encodings are concatenated and passed through a fully connected layer.
- Output layer: The result of the modeling layer is passed to two independent FCL + softmax networks to compute the begin and end span distribution of the predicted answer. 
This is implemented in `qa_baseline_model.py`.
* Bidaf: We replaced the basic attention layer by a bidirectional attention layer. This is implemented in `qa_bidaf_model.py`.

* Self Attention: We added a self attention layer on top of the basic attention layer from the baseline. This is implemented in `qa_selfattn_model.py`. 

* Stack: Self attention layer was added on top of the bidirectional attention layer and the modeling layer is replaced by a bidirectional RNN layer. This is implemented in `qa_stack_model.py`.

* Pointer: We introduce a dependency between the end span distribution and begin span distribution. This is implemented in `qa_pointer_model.py`.

### Training 
To launch training
```
python code/main.py --mode train
``` 

For training the following arguments can be used to choose which model to train.
* `--model_name`: baseline/bidaf/selfattn/stack/pointer
* `--rnn_cell`: Decides whether to use GRU or LSTM units for the encoding and modeling layers.
* `--num_layers`: Decides how many RNN layers to use for encoding.
* `--selfattn_size`: Decides the hidden size for the self attention layer
* `--hidden_size`: Decides the hidden size for the RNN units. 
Other options for training including learning rates, dropout and gradient clipping can be found in `code/main.py`. 


## Results & Evaluation
Evaluation and ensembling of models was done on Codalab using the `codalab_upload.sh` script.
Local evaluation for one model can be done using the `official_eval.sh` script. (Note that if non-default options are used during training, these options need to also be added in the call for `code/main.py`). 


|Model | EM Score | F1 Score |
| ------------- |:-------------:| -----:|
|Baseline (hidden size 200) |34.38 |43.622 |
|BiDAF (hidden size 200) | 40.25 | 50.04|
|Self Attention | 51.81 | 65.12 
|BiDAF + SelfAttention | 58.54 | 69.40 |
|Stack-100 | 60.96 | 71.63 |
|Stack-100 + modeling layer |	60.7 | 71.60 |
|Ensemble | 65.025 |75.249 | 

## Attention visualization 
Once a model is trained, it is possible to visualize the attention distributions computed on some examples as well as the begin and end probability distributions. 
To do so 
```
python code/main.py --mode visualize --train_dir path/to/checkpoint [ADD ALL OTHER OPTIONS USED FOR TRAINING]
```
This will compute the distributions on the dev set and save them as `.npy` files in the specified training directory. 
To generate the visualization images 
```
python code/utils/result_analysis.py
``` 
/!\ The target directory where saved `.npy` files are needs to be specified manually in the code. 
This script will perform some basic result analysis (F1/EM scores, breakdown stats by question and context types) and generate attention visualizations. 

![Context To Question Attention](https://user-images.githubusercontent.com/13089230/38461067-e6ca51e4-3a7b-11e8-9113-bf23043e9e65.png)

## References 
 * GloVe: [website](https://nlp.stanford.edu/projects/glove/)
Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
* SQuAD: [website](https://rajpurkar.github.io/SQuAD-explorer/)
Rajpurkar, Pranav, et al. "Squad: 100,000+ questions for machine comprehension of text." arXiv preprint arXiv:1606.05250 (2016).
* Seo, Minjoon, et al. "Bidirectional attention flow for machine comprehension." arXiv preprint arXiv:1611.01603 (2016).
* Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm and answer pointer." arXiv preprint arXiv:1608.07905 (2016).

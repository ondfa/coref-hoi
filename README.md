This repo is a codebase snapshot of [lxucs/coref-hoi](https://github.com/lxucs/coref-hoi); active issues or updates are maintained in [lxucs/coref-hoi](https://github.com/lxucs/coref-hoi) repository.

# End-to-End Coreference Resolution with Different Higher-Order Inference Methods

This repository contains the implementation of the paper: [Multilingual Coreference Resolution with Harmonized Annotations](https://aclanthology.org/2021.ranlp-1.125) based on [Revealing the Myth of Higher-Order Inference in Coreference Resolution](https://www.aclweb.org/anthology/2020.emnlp-main.686.pdf).

## Architecture

The basic end-to-end coreference model is a PyTorch re-implementation based on the TensorFlow model following similar preprocessing (see this [repository](https://github.com/mandarjoshi90/coref)).

**Files**:
* [run.py](run.py): training and evaluation
* [model.py](model.py): the coreference model
* [higher_order.py](higher_order.py): higher-order inference modules
* [analyze.py](analyze.py): result analysis
* [preprocess.py](preprocess.py): converting CoNLL files to examples
* [tensorize.py](tensorize.py): tensorizing example
* [conll.py](conll.py), [metrics.py](metrics.py): same CoNLL-related files from the [repository](https://github.com/mandarjoshi90/coref)
* [experiments.conf](experiments.conf): different model configurations

## Basic Setup
Set up environment and data for training and evaluation:
* Install Python3 dependencies: `pip install -r requirements.txt`
* Create a directory for data that will contain all data files, models and log files; set `data_dir = /path/to/data/dir` in [experiments.conf](experiments.conf)
* Prepare dataset (requiring [CorefUD](https://ufallab.ms.mff.cuni.cz/~popel/CorefUD-1.0-public.zip) corpus):
* `python preprocess.py [config]`
  * e.g. `python preprocess.py train_mbert_czech`

## Evaluation

The name of each directory corresponds with a **configuration** in [experiments.conf](experiments.conf). Each directory has two trained models inside.

If you want to use the official evaluator, download and unzip [corefUD scorer](https://cs.emory.edu/~lxu85/conll-2012.zip) under this directory.

Evaluate a model on the dev/test set:
* Download the corresponding model directory and unzip it under `data_dir`
* `python evaluate.py [config] [model_id] [gpu_id]`
    * e.g. Attended Antecedent:`python evaluate.py train_spanbert_large_ml0_d2 May08_12-38-29_58000 0`

## Training
`python run.py [config] [gpu_id]`

* [config] can be any **configuration** in [experiments.conf](experiments.conf)
* Log file will be saved at `your_data_dir/[config]/log_XXX.txt`
* Models will be saved at `your_data_dir/[config]/model_XXX.bin`
* Tensorboard is available at `your_data_dir/tensorboard`


## Configurations
Some important configurations in [experiments.conf](experiments.conf):
* `data_dir`: the full path to the directory containing dataset, models, log files
* `bert_pretrained_name_or_path`: the name/path of the pretrained BERT model ([HuggingFace BERT models](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.from_pretrained))
* `max_training_sentences`: the maximum segments to use when document is too long.

## Results

|                | F1    | F1 (without singletons) |
|----------------|-------|-------------------------|
| catalan        | 50.29 | 62.78                   |
| czech          | 60.52 | 66.64                   |
| czech-pcedt    | 69.59 | 69.73                   |
| english-gum    | 50.80 | 65.76                   |
| english-parcor | 57.47 | 58.12                   |
| german         | 45.35 | 58.89                   |
| german-parcor  | 55.40 | 56.51                   |
| hungarian      | 56.15 | 57.40                   |
| lithuanian     | 67.02 | 67.90                   |
| polish         | 43.13 | 62.39                   |
| russian        | 62.33 | 62.43                   |
| spanish        | 50.22 | 64.81                   |
|                |
| avg            | 54.19 | 62.48                   |

## Citation
```
@inproceedings{pravzak2021multilingual,
  title={Multilingual Coreference Resolution with Harmonized Annotations},
  author={Pra{\v{z}}{\'a}k, Ond{\v{r}}ej and Konop{\'\i}k, Miloslav and Sido, Jakub},
  booktitle={Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021)},
  pages={1119--1123},
  year={2021}
}
```
# Assessing transformer models with scalar adjective expressions
This is the code and dataset repository for 'Assessing transformer models with scalar adjective expressions'.

## Datasets

Datasets including scalar diversity dataset by Gotzner et al. (2018) (ng_cleaned.csv), scalar adjective dataset used for finetuning (si_100.txt, si_500.txt), naturalistic test data for context insentivity of scalar adjective implicatures (post.json) can be found in the dataset folder.

## Code

### 1. If you want to reproduce the dataset collection steps, use the relevant code in ```prepare_dataset```.

You can use codes in ```generate_implicature``` to filter reddit comments from [Politosphere (Hofmann et al., 2022)](https://ojs.aaai.org/index.php/ICWSM/article/view/19377) and generate corresponding scalar implicatures.
First navigate to this folder, then run 
```
python extract_posts.py --out_path [YOUR OUTPUT PATH] --device [YOUR DEVICE]
```
to extract posts for scalar adjective pairs.
Then run
```
python generate_implicature.py --post_path [YOUR PATH FOR EXTRACTED POSTS] --outpath [YOUR OUTPUT PATH]
```
to generate implicatures for posts.

To generate dataset for continued training, navigate to the folder ```select_post_for_mlm_training```.
Then run
```python select_post_for_training.py --out_path [YOUR OUTPUT PATH]```
to get si_100.txt, si_500.txt use for training.

### 2. If you want to reproduce results for scale intensity ranking, refer to [this repo (Garí Soler & Apidianaki, 2020)](https://github.com/ainagari/scalar_adjs)
Scalar intensity ranking in this work directly uses code provided in this repo.

### 3. If you want to reproduce results for scale alignment, use codes in ```scale_alignment``` folder.
Part of code in this folder is adapted from [this repo (Garí Soler & Apidianaki, 2020)](https://github.com/ainagari/scalar_adjs). 

First, you need to generate contextualised embeddings for models. Download scales and context sentences from [here](https://github.com/ainagari/scalar_adjs/tree/master/data)(Garí Soler & Apidianaki, 2020), use ```ukwac_selected_scalar_sentences.pkl``` as context sentences.
Run
```
python adj_embedding_generation.py --sentence_path [YOUR SENTENCE FILE PATH] --out_path [YOUR OUTPUT PATH] --model [BERT or RoBERTa] --size [BASE OR LARGE]
--device [YOUR DEVICE]
```
To get results for distinguishing in-scale adjs from random ones, run
```
python adj_scale_alignment.py --embedding_path [YOUR EMBEDDING FOLDER] --out_path [YOUR OUTPUT PATH]
```
To get results for fine-grained scale alignment, run
```
python adj_scale_classification.py --term_path [YOUR ADJ TERM FOLDER]] --embedding_path [YOUR EMBEDDING FOLDER] --out_path [YOUR OUTPUT PATH]
```
To get results for fine-grained scale alignment after pruning, run
```
python adj_scale_classification_pruned.py --term_path [YOUR ADJ TERM FOLDER] --embedding_path [YOUR EMBEDDING FOLDER] --out_path [YOUR OUTPUT PATH] --prune_freq [TRUE OR FALSE] --prune_overlap [TRUE OR FALSE]
```

### 4. If you want to reproduce results for scalar implicature experiments, use codes provided in ```scalar_implicature_diversity``` folder.
To reproduce results for controlled experiments, first navigate to the ```controlled``` folder.
Then run
```
python zeroshot_mlm.py --out_path [YOUR OUTPUT PATH] --device [YOUR DEVICE]
```
to get vanilla implicature results in a zero-shot setting.
If you want to get results after training models with more unlabelled data (si-100.txt/si-500.txt), first run
```
python further_training.py --dataset_path [PATH FOR DATASET] --model_name [MODEL NAME] --out_path [YOUR OUTPUT PATH] --device [YOUR DEVICE]
```
to get trained model.
Then, run
```
python naturalistic.py --post_path [YOUR PATH FOR POST AND IMPLICATURES] --out_path [YOUR OUTPUT PATH] --device [YOUR DEVICE]
```

from __future__ import division
from __future__ import print_function
import os
import argparse
from typing import Union, Dict, Optional, Any, List

# Rafiki Dependency
from rafiki.model import PandaModel, FloatKnob, CategoricalKnob, FixedKnob, IntegerKnob, PolicyKnob, utils
from rafiki.model.knob import BaseKnob
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

# PyTorch Dependency
import torch.nn as nn
from torchvision.models.vgg import vgg11_bn

# Misc Third-party Machine-Learning Dependency
import numpy as np

# Panda Modules Dependency
from rafiki.panda.models.PandaTorchBasicModel import PandaTorchBasicModel


import numpy as np
import pandas as pd
import csv
import glob
import json
import re
import pickle
import torch
import semanticscholar as sch
from IPython.display import display, Latex, HTML, FileLink
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel


KnobConfig = Dict[str, BaseKnob]
Knobs = Dict[str, Any]
Params = Dict[str, Union[str, int, float, np.ndarray]]

root_path = "data/covid19data"
metadata_path = "{}/metadata.csv".format(root_path)
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})
# meta_df.head()
all_json = glob.glob("{}/**/*.json".format(root_path), recursive=True)

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except:
                self.abstract.append("No abstract available")
            for entry in content["body_text"]:
                self.body_text.append(entry['text'])
            self.abstract = '. '.join(self.abstract)
            self.body_text = '. '.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[1])

dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'publish_time': [],
         'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)

    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue

    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)

    try:
        authors = meta_data['authors'].values[0].split(';')
        dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if Null value
        dict_['authors'].append(meta_data['authors'].values[0])

    # add the title information
    dict_['title'].append(meta_data['title'].values[0])

    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])

    # add the publishing data
    dict_['publish_time'].append(meta_data['publish_time'].values[0])

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'publish_time'])
df_covid.drop_duplicates(['title'], inplace=True)
df_covid.dropna(subset=['body_text'], inplace=True)

covid_terms =['covid', 'coronavirus disease 19', 'sars cov 2', '2019 ncov', '2019ncov', '2019 n cov', '2019n cov',
              'ncov 2019', 'n cov 2019', 'coronavirus 2019', 'wuhan pneumonia', 'wuhan virus', 'wuhan coronavirus',
              'coronavirus 2', 'covid-19', 'SARS-CoV-2', '2019-nCov']
covid_terms = [elem.lower() for elem in covid_terms]
covid_terms = re.compile('|'.join(covid_terms))

def checkYear(date):
    return int(date[0:4])

def checkCovid(row, covid_terms):
    return bool(covid_terms.search(row['body_text'].lower())) and checkYear(row['publish_time']) > 2019

df_covid['is_covid'] = df_covid.apply(checkCovid, axis=1, covid_terms=covid_terms)
df_covid_only = df_covid[df_covid['is_covid']==True]
df_covid_only = df_covid_only.reset_index(drop=True)

def preprocessing(text):
    # remove mail
    text = re.sub(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', 'MAIL', text)
    # remove doi
    text = re.sub(r'https\:\/\/doi\.org[^\s]+', 'DOI', text)
    # remove https
    text = re.sub(r'(\()?\s?http(s)?\:\/\/[^\)]+(\))?', '\g<1>LINK\g<3>', text)
    # remove single characters repeated at least 3 times for spacing error (e.g. s u m m a r y)
    text = re.sub(r'(\w\s+){3,}', ' ', text)
    # replace tags (e.g. [3] [4] [5]) with whitespace
    text = re.sub(r'(\[\d+\]\,?\s?){3,}(\.|\,)?', ' \g<2>', text)
    # replace tags (e.g. [3, 4, 5]) with whitespace
    text = re.sub(r'\[[\d\,\s]+\]', ' ',text)
     # replace tags (e.g. (NUM1) repeated at least 3 times with whitespace
    text = re.sub(r'(\(\d+\)\s){3,}', ' ',text)
    # replace '1.3' with '1,3' (we need it for split later)
    text = re.sub(r'(\d+)\.(\d+)', '\g<1>,\g<2>', text)
    # remove all full stops as abbreviations (e.g. i.e. cit. and so on)
    text = re.sub(r'\.(\s)?([^A-Z\s])', ' \g<1>\g<2>', text)
    # correctly spacing the tokens
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    # return lowercase text
    return text.lower()

df_covid_only['preproc_body_text'] = df_covid_only['body_text'].apply(preprocessing)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def checkAnyStop(token_list, token_stops):
    return any([stop in token_list for stop in token_stops])

def firstFullStopIdx(token_list, token_stops):
    """
    Returns the index of first full-stop token appearing.
    """
    idxs = []
    for stop in token_stops:
        if stop in token_list:
            idxs.append(token_list.index(stop))
    minIdx = min(idxs) if idxs else None
    return minIdx

puncts = ['!', '.', '?', ';']
puncts_tokens = [tokenizer.tokenize(x)[0] for x in puncts]

def splitTokens(tokens, punct_tokens, split_length):
    """
    To avoid splitting a sentence and lose the semantic meaning of it, a paper is splitted
    into chunks in such a way that each chunk ends with a full-stop token (['.' ';' '?' or '!'])
    """
    splitted_tokens = []
    while len(tokens) > 0:
        if len(tokens) < split_length or not checkAnyStop(tokens, punct_tokens):
            splitted_tokens.append(tokens)
            break
        # to not have too long parapraphs, the nearest fullstop is searched both in the previous
        # and the next strings.
        prev_stop_idx = firstFullStopIdx(tokens[:split_length][::-1], puncts_tokens)
        next_stop_idx = firstFullStopIdx(tokens[split_length:], puncts_tokens)
        if pd.isna(next_stop_idx):
            splitted_tokens.append(tokens[:split_length - prev_stop_idx])
            tokens = tokens[split_length - prev_stop_idx:]
        elif pd.isna(prev_stop_idx):
            splitted_tokens.append(tokens[:split_length + next_stop_idx + 1])
            tokens = tokens[split_length + next_stop_idx + 1:]
        elif prev_stop_idx < next_stop_idx:
            splitted_tokens.append(tokens[:split_length - prev_stop_idx])
            tokens = tokens[split_length - prev_stop_idx:]
        else:
            splitted_tokens.append(tokens[:split_length + next_stop_idx + 1])
            tokens = tokens[split_length + next_stop_idx + 1:]
    return splitted_tokens

def splitParagraph(text, split_length=90):
    tokens = tokenizer.tokenize(text)
    splitted_tokens = splitTokens(tokens, puncts_tokens, split_length)
    return [tokenizer.convert_tokens_to_string(x) for x in splitted_tokens]

df_covid_only['body_text_parags'] = df_covid_only['preproc_body_text'].apply(splitParagraph)

text = df_covid_only['body_text_parags'].to_frame()
body_texts = text.stack().tolist()

encoding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

covid_encoded = []
for body in tqdm(body_texts):
    covid_encoded.append(encoding_model.encode(body, show_progress_bar=False))

def computeMaxCosine(encoded_query, encodings):
    cosines = cosine_similarity(encoded_query[0].reshape(1, -1), encodings)
    return float(np.ndarray.max(cosines, axis=1))


def extractPapersIndexes(query, num_papers=5):
    encoded_query = encoding_model.encode([query.replace('?', '')], show_progress_bar=False)
    cosines_max = []
    for idx in range(len(covid_encoded)):
        paper = np.array(covid_encoded[idx])
        result = computeMaxCosine(encoded_query, paper)
        cosines_max.append(result)

    indexes_max_papers = np.array(cosines_max).argsort()[-num_papers:][::-1]
    return indexes_max_papers


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD)
qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD)

qa_model = qa_model.to(torch_device)
qa_model.eval()

def findStartEndIndexSubstring(context, answer):
    """
    Search of the answer inside the paragraph. It returns the start and end index.
    """
    search_re = re.search(re.escape(answer.lower()), context.lower())
    if search_re:
        return search_re.start(), search_re.end()
    else:
        return 0, len(context)

def postprocessing(text):
    # capitalize the text
    text = text.capitalize()
    # '2 , 3' -> '2,3'
    text = re.sub(r'(\d) \, (\d)', "\g<1>,\g<2>", text)
    # full stop
    text = re.sub(r' (\.|\!|\?) (\w)', lambda pat: pat.groups()[0]+" "+pat.groups()[1].upper(), text)
    # full stop at the end
    text = re.sub(r' (\.|\!|\?)$', "\g<1>", text)
    # comma
    text = re.sub(r' (\,|\;|\:) (\w)', "\g<1> \g<2>", text)
    # - | \ / @ and _ (e.g. 2 - 3  -> 2-3)
    text = re.sub(r'(\w) (\-|\||\/|\\|\@\_) (\w)', "\g<1>\g<2>\g<3>", text)
    # parenthesis
    text = re.sub(r'(\(|\[|\{) ([^\(\)]+) (\)|\]|\})', "\g<1>\g<2>\g<3>", text)
    # "" e.g. " word "  -> "word"
    text = re.sub(r'(\") ([^\"]+) (\")', "\g<1>\g<2>\g<3>", text)
    # apostrophe  e.g. l ' a  -> l'a
    text = re.sub(r'(\w) (\') (\w)', "\g<1>\g<2>\g<3>", text)
    # '3 %' ->  '3%'
    text = re.sub(r'(\d) \%', "\g<1>%", text)
    # '# word'  -> '#word'
    text = re.sub(r'\# (\w)', "#\g<1>", text)
    # https and doi
    text = re.sub(r'(https|doi) : ', "\g<1>:", text)
    return text

keywords_list = []
idx_design_map = []
with open("/kaggle/input/loe-keywords/loe_keywords.tsv") as infile:
    tsvreader = csv.reader(infile, delimiter="\t")
    for idx, line in enumerate(tsvreader):
        if idx == 0 or idx == 1:
            continue
        keywords_list.append(line[2].split(", "))
        idx_design_map.append(line[0])

keywords_list = [list(filter(None, elem)) for elem in keywords_list]
minimum_occurrencies = [2, 2, 2, 1, 4, 3, 3, 1, 1]

def getSingleLOEScore(paper_idx, loe_idx):
    keywords_analysed = re.compile("|".join(keywords_list[loe_idx]))
    return len(re.findall(keywords_analysed, df_covid_only['body_text'][paper_idx]))

def getPaperLOE(paper_idx):
    title_search = re.compile("systematic review|meta-analysis")
    if title_search.search(df_covid_only['title'][paper_idx]):
        return "systematic review and meta-analysis"
    scores = [getSingleLOEScore(paper_idx, idx) for idx in range(len(idx_design_map))]
    sorted_indexes = np.argsort(scores)[::-1]
    for sorted_index in sorted_indexes:
        if scores[sorted_index] >= minimum_occurrencies[sorted_index]:
            return idx_design_map[sorted_index]
    return "Information not available"

'''def getFullDictQueries(data, num_papers=5): # predict
    final_dict = {}
    for task in data.keys():
        print(f"Processing: {task}")
        task_questions = data[task]['questions']
        dict_task_quest ={}
        dict_queries = {}
        for idx, query in enumerate(task_questions):
            print(f"Getting answers from query: {query}")
            indexes_papers = extractPapersIndexes(query, num_papers=num_papers)
            dict_queries[query] = getAllContexts(query, indexes_papers)
        dict_task_quest['queries'] = dict_queries
        dict_task_quest['task_name'] = data[task]['area']
        final_dict[task] = dict_task_quest
    return final_dict'''


def answerQuestion(question, paper):
    """
    This funtion provides the best answer found by the Q&A model, the chunk containing it
    among all chunks of the input paper and the score obtained by the answer
    """
    inputs = [qa_tokenizer.encode_plus(
        question, paragraph, add_special_tokens=True, return_tensors="pt") for paragraph in paper]
    answers = []
    confidence_scores = []
    for n, Input in enumerate(inputs):
        input_ids = Input['input_ids'].to(torch_device)
        token_type_ids = Input['token_type_ids'].to(torch_device)
        if len(input_ids[0]) > 510:
            input_ids = input_ids[:, :510]
            token_type_ids = token_type_ids[:, :510]
        text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids[0])
        start_scores, end_scores = qa_model(input_ids, token_type_ids=token_type_ids)
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        # if the start token of the answer is contained in the question, the start token is moved to
        # the first one of the chunk
        check = text_tokens.index("[SEP]")
        if int(answer_start) <= check:
            answer_start = check + 1
        answer = qa_tokenizer.convert_tokens_to_string(text_tokens[answer_start:(answer_end + 1)])
        answer = answer.replace('[SEP]', '')
        confidence = start_scores[0][answer_start] + end_scores[0][answer_end]
        if answer.startswith('. ') or answer.startswith(', '):
            answer = answer[2:]
        answers.append(answer)
        confidence_scores.append(float(confidence))

    maxIdx = np.argmax(confidence_scores)
    confidence = confidence_scores[maxIdx]
    best_answer = answers[maxIdx]
    best_paragraph = paper[maxIdx]

    return best_answer, confidence, best_paragraph

class PyPandaQA(PandaTorchBasicModel):
    """
    Implementation of PyTorch Vgg
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)

    def _create_model(self, scratch: bool, num_classes: int):

        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"
        qa_tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD)

        qa_model = qa_model.to(torch_device)
        qa_model.eval()
        # print("create model {}".format(qa_model))
        return qa_model


    def train(self):
        pass
    def evaluat(self):
        pass
    def predict(self, queries=None): ### file location --> (data, num_papers=5) --> (question, df_covid_only['body_text_parags'][paper_index])
        num_papers=5
        if queries is None:
            data = {"Task1":
                        {'area': 'What is known about transmission, incubation, and environmental stability?',
                         'questions': ['What is the range of the incubation period in humans?',
                                       'How long individuals are contagious?',
                                       'How long does the virus persist on surfaces?',
                                       'What is the natural history of the virus?',
                                       'Are there diagnostics to improve clinical processes?',
                                       'What is known about immunity?',
                                       'Are movement control strategies effective?',
                                       'Is personal protective equipment effective?',
                                       'Does the environment affect transmission?']},
                    "Task2":
                        {'area': 'What do we know about COVID-19 risk factors?',
                         'questions': ['Is smoking a risk factor?',
                                       'Are pulmunary diseases risk factors?',
                                       'Are co-infections risk factors?',
                                       'What is the basic reproductive number?',
                                       'What is the serial interval?',
                                       'Which are the environmental risk factors?',
                                       'What is the severity of the disease?',
                                       'Which are high-risk patient groups?',
                                       'Are there public health mitigation measures?'
                                       ]}
                    }
        else:
            data = queries # as input

        final_dict = {}
        for task in data.keys():
            print(f"Processing: {task}")
            task_questions = data[task]['questions']
            dict_task_quest = {}
            dict_queries = {}
            for idx, query in enumerate(task_questions):
                print(f"Getting answers from query: {query}")
                indexes_papers = extractPapersIndexes(query, num_papers=num_papers)
                # getAllContexts()
                (question, indexes_papers)=(query, indexes_papers)
                answers_list = []
                for paper_index in indexes_papers:
                    answer, conf, paragraph = answerQuestion(question, df_covid_only['body_text_parags'][paper_index])  # best_answer, confidence, best_paragraph
                    if answer:
                        author = df_covid_only['authors'][paper_index] if not pd.isna(
                            df_covid_only['authors'][paper_index]) else "not available"
                        journal = df_covid_only['journal'][paper_index] if not pd.isna(
                            df_covid_only['journal'][paper_index]) else "not available"
                        title = df_covid_only['title'][paper_index] if not pd.isna(
                            df_covid_only['title'][paper_index]) else "not available"
                        start, end = findStartEndIndexSubstring(paragraph, answer)
                        answer_parag = getSingleContext(paragraph, start, end)
                        paper_citations_count, journal_api = getAPIInformations(df_covid_only['paper_id'][paper_index])
                        journal = journal_api if journal_api else journal
                        journal_row = scimago_jr[scimago_jr['Title'].apply(lambda x: x.lower()) == journal.lower()]
                        journal_score = journal_row[
                            'SJR'].item() if not journal_row.empty else "Information not available"
                        loe = getPaperLOE(paper_index)
                        paper_answer = {
                            "answer": answer_parag,
                            "title": title,
                            "journal": journal,
                            "author": author,
                            "journal_score": journal_score,
                            "paper_citations_count": paper_citations_count,
                            "level_of_evidence": loe
                        }
                        answers_list.append(paper_answer)
                dict_queries[query] = answers_list
            dict_task_quest['queries'] = dict_queries
            dict_task_quest['task_name'] = data[task]['area']
            final_dict[task] = dict_task_quest

        return final_dict
'''
    @staticmethod
    def get_knob_config():
        return {
            'model_class':CategoricalKnob(['vgg_cifar10']),
            # Learning parameters
            'lr':FixedKnob(0.0001), ### learning_rate
            'weight_decay':FixedKnob(0.0),
            'drop_rate':FixedKnob(0.0),
            'max_epochs': FixedKnob(30), 
            'batch_size': CategoricalKnob([200]),
            'max_iter': FixedKnob(20),
            'optimizer':CategoricalKnob(['adam']),
            'scratch':FixedKnob(True),

            # Data augmentation
            'max_image_size': FixedKnob(32),
            'share_params': CategoricalKnob(['SHARE_PARAMS']),
            'tag':CategoricalKnob(['relabeled']),
            'workers':FixedKnob(8),
            'seed':FixedKnob(123456),
            'scale':FixedKnob(512),
            'horizontal_flip':FixedKnob(True),
     
            # Hyperparameters for PANDA modules
            # Self-paced Learning and Loss Revision
            'enable_spl':FixedKnob(False),
            'spl_threshold_init':FixedKnob(16.0),
            'spl_mu':FixedKnob(1.3),
            'enable_lossrevise':FixedKnob(False),
            'lossrevise_slop':FixedKnob(2.0),

            # Label Adaptation
            'enable_label_adaptation':FixedKnob(False), # error occurs 

            # GM Prior Regularization
            'enable_gm_prior_regularization':FixedKnob(False),
            'gm_prior_regularization_a':FixedKnob(0.001),
            'gm_prior_regularization_b':FixedKnob(0.0001),
            'gm_prior_regularization_alpha':FixedKnob(0.5),
            'gm_prior_regularization_num':FixedKnob(4),
            'gm_prior_regularization_lambda':FixedKnob(0.0001),
            'gm_prior_regularization_upt_freq':FixedKnob(100),
            'gm_prior_regularization_param_upt_freq':FixedKnob(50),
            
            # Explanation
            'enable_explanation': FixedKnob(False),
            'explanation_gradcam': FixedKnob(True),
            'explanation_lime': FixedKnob(False),

            # Model Slicing
            'enable_model_slicing':FixedKnob(False),
            'model_slicing_groups':FixedKnob(0),
            'model_slicing_rate':FixedKnob(1.0),
            'model_slicing_scheduler_type':FixedKnob('randomminmax'),
            'model_slicing_randnum':FixedKnob(1),

            # MC Dropout
            'enable_mc_dropout':FixedKnob(False),
            'mc_trials_n':FixedKnob(10)
        }'''

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/cifar10_train.zip', help='Path to train dataset')
    parser.add_argument('--val_path', type=str, default='data/cifar10_test.zip', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='data/cifar10_val.zip', help='Path to test dataset')
    print (os.getcwd())
    parser.add_argument(
        '--query_path', 
        type=str, 
        default=
        # 'examples/data/image_classification/1463729893_339.jpg,examples/data/image_classification/1463729893_326.jpg,examples/data/image_classification/eed35e9d04814071.jpg',
        'examples/data/image_classification/1463729893_339.jpg',
        help='Path(s) to query image(s), delimited by commas')'''
    # (args, _) = parser.parse_known_args()

    # queries = utils.dataset.load_images(args.query_path.split(',')).tolist()
    
    test_model_class(
        model_file_path=__file__,
        model_class='PyPandaQA',
        task='IMAGE_CLASSIFICATION',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            # ModelDependency.TORCHVISION: '0.2.2',
            # ModelDependency.CV2: '4.2.0.32'
        },
        # train_dataset_path=args.train_path,
        # val_dataset_path=args.val_path,
        # test_dataset_path=args.test_path,
        # queries=queries
    )

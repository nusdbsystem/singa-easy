from __future__ import division
from __future__ import print_function
import argparse
from typing import Union, Dict, Optional, Any, List

from singa_auto.model import CategoricalKnob, FixedKnob, utils, BaseModel
from singa_auto.model.knob import BaseKnob
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class

# PyTorch Dependency
import torch

# Misc Third-party Dependency
import numpy as np
import pandas as pd
import csv
import glob
import json
import re
import pickle
import semanticscholar as sch
from IPython.display import display, Latex, HTML, FileLink
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
from bs4 import BeautifulSoup

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


all_json = glob.glob("{}/**/*.json".format(root_path), recursive=True)
# all_json = glob.glob("{}/arxiv/**/*.json".format(root_path), recursive=True)

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

scimago_jr = pd.read_csv('data/covid19data/scimagojr_2018.csv', sep=';')
scimago_jr.drop_duplicates(['Title'], inplace=True)
scimago_jr = scimago_jr.reset_index(drop=True)

def getAPIInformations(paper_id):
    paper_api = sch.paper(paper_id)
    if paper_api:
        return paper_api['influentialCitationCount'], paper_api['venue']
    else:
        return "Information not available", None

def getSingleContext(context, start, end):
    before_answer = context[:start]
    answer = context[start:end]
    after_answer = context[end:]
    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer
    context_answer = """<div class="single_answer">{}</div>""".format(postprocessing(content))
    return context_answer

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
    print (best_answer, confidence, best_paragraph)
    return best_answer, confidence, best_paragraph

class Question_Answering(BaseModel):
    """
    Implementation of QuestionAnswering model
    """
    def __init__(self, **knobs):
        super().__init__(**knobs)

    def _create_model(self, scratch: bool):
        # if scratch == False:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"
        qa_tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD)

        qa_model = qa_model.to(torch_device)
        qa_model.eval()
        return qa_model


    def train(self, dataset_path=None, shared_params: Optional[Params] = None, **train_args):
        self._model = self._create_model(scratch = False)
        num_papers = 5

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

        final_dict = {}
        for task in data.keys():
            print(f"Processing: {task}")
            task_questions = data[task]['questions']
            dict_task_quest = {}
            dict_queries = {}
            for idx, query in enumerate(task_questions):
                print(f"Getting answers from query: {query}")
                indexes_papers = extractPapersIndexes(query, num_papers=num_papers)

                (question, indexes_papers) = (query, indexes_papers)
                answers_list = []
                for paper_index in indexes_papers:
                    # best_answer, confidence, best_paragraph
                    answer, conf, paragraph = answerQuestion(question, df_covid_only['body_text_parags'][
                        paper_index])  
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
                        paper_answer = {
                            "answer": answer_parag,
                            "title": title,
                            "journal": journal,
                            "author": author,
                            "journal_score": journal_score,
                            "paper_citations_count": paper_citations_count,
                        }
                        answers_list.append(paper_answer)
                dict_queries[query] = answers_list
            dict_task_quest['queries'] = dict_queries
            dict_task_quest['task_name'] = data[task]['area']
            final_dict[task] = dict_task_quest

        full_code = final_dict

        cnt = 1
        task_cnt = 1
        for task_key in full_code.keys():
            results_table = pd.DataFrame(columns=['Question', 'Title', 'Authors', 'Answer', 'Journal', 'Journal score',
                                                  'Paper citations count', ])
            for question_key in full_code[task_key]['queries'].keys():
                for idx in range(len(full_code[task_key]['queries'][question_key])):
                    row = [question_key,
                           full_code[task_key]['queries'][question_key][idx]['title'],
                           full_code[task_key]['queries'][question_key][idx]['author'],
                           BeautifulSoup(full_code[task_key]['queries'][question_key][idx]['answer']).text,
                           full_code[task_key]['queries'][question_key][idx]['journal'],
                           full_code[task_key]['queries'][question_key][idx]['journal_score'],
                           full_code[task_key]['queries'][question_key][idx]['paper_citations_count'],
                           ]
                    results_table.loc[cnt] = row
                    cnt += 1
            # save as output
            results_table.to_csv("test/task_{}_results.tsv".format(task_cnt))
            task_cnt += 1
        print(results_table, task_cnt)
        return results_table

    def evaluate(self, dataset_path):
        return float(1)

    def predict(self, queries=None): 
        pass
    

    def dump_parameters(self):
        params = {
        }

        return params

    def load_parameters(self, params):
        self._model = self._create_model()

    @staticmethod
    def get_knob_config():
        return {
            'model_class':CategoricalKnob(['question_answering']),}


if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='Question_Answering',
        task='Question_Answering', 
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
        },
        train_dataset_path='',
        val_dataset_path='',
        queries=None
    )

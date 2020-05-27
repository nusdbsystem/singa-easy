from __future__ import division
from __future__ import print_function
import argparse
from typing import Union, Dict, Optional, Any

from singa_auto.model import CategoricalKnob, FixedKnob, utils, BaseModel
from singa_auto.model.knob import BaseKnob
from singa_auto.constants import ModelDependency
from singa_auto.model.dev import test_model_class

# PyTorch Dependency
import torch

# Misc Third-party Dependency
import numpy as np
import pandas as pd
import glob
import json
import re
import semanticscholar as sch
from IPython.display import display, Latex, HTML, FileLink
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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

# all_json = glob.glob("{}/**/*.json".format(root_path), recursive=True)
all_json = glob.glob("{}/arxiv/**/*.json".format(root_path), recursive=True)

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

BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad");
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

def computeMaxCosine(encoded_query, encodings):
    cosines = cosine_similarity(encoded_query[0].reshape(1, -1), encodings)
    return float(np.ndarray.max(cosines, axis=1))
encoding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens');
covid_encoded = []
for body in tqdm(body_texts):
    covid_encoded.append(encoding_model.encode(body, show_progress_bar=False))
def extractPapersIndexes(query, num_papers=5):
    encoded_query = encoding_model.encode([query.replace('?', '')], show_progress_bar=False)
    cosines_max = []
    for idx in range(len(covid_encoded)):
        paper = np.array(covid_encoded[idx])
        result = computeMaxCosine(encoded_query, paper)
        cosines_max.append(result)

    indexes_max_papers = np.array(cosines_max).argsort()[-num_papers:][::-1]
    return indexes_max_papers

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

def getSingleContext(context, start, end):
    before_answer = context[:start]
    answer = context[start:end]
    after_answer = context[end:]
    content = before_answer + "<span class='answer'>" + answer + "</span>" + after_answer
    context_answer = """<div class="single_answer">{}</div>""".format(postprocessing(content))
    return context_answer

def findStartEndIndexSubstring(context, answer):
    """
    Search of the answer inside the paragraph. It returns the start and end index.
    """
    search_re = re.search(re.escape(answer.lower()), context.lower())
    if search_re:
        return search_re.start(), search_re.end()
    else:
        return 0, len(context)

def getAPIInformations(paper_id):
    paper_api = sch.paper(paper_id)
    if paper_api:
        return paper_api['influentialCitationCount'], paper_api['venue']
    else:
        return "Information not available", None
qa_tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD);
def answerQuestion(question, paper, qa_model):
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

scimago_jr = pd.read_csv('data/covid19data/scimagojr_2018.csv', sep=';')
scimago_jr.drop_duplicates(['Title'], inplace=True)
scimago_jr = scimago_jr.reset_index(drop=True)
def QA(queries: dict, qa_model, num_papers = 5):
        data = queries[0]
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
                    answer, conf, paragraph = answerQuestion(question, df_covid_only['body_text_parags'][
                        paper_index], qa_model)  # best_answer, confidence, best_paragraph
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
        return getHtmlCode(full_code, layoutStyle())
        # # to return a table
        # cnt = 1
        # for task_key in full_code.keys():
        #     results_table = pd.DataFrame(columns=['Question', 'Title', 'Authors', 'Answer', 'Journal', 'Journal score',
        #                                           'Paper citations count', ])
        #     # iterate across questions for each task
        #     for question_key in full_code[task_key]['queries'].keys():
        #         # idx is the num_papers specified
        #         for idx in range(len(full_code[task_key]['queries'][question_key])):
        #             row = [question_key,
        #                    full_code[task_key]['queries'][question_key][idx]['title'],
        #                    full_code[task_key]['queries'][question_key][idx]['author'],
        #                    BeautifulSoup(full_code[task_key]['queries'][question_key][idx]['answer']).text,
        #                    full_code[task_key]['queries'][question_key][idx]['journal'],
        #                    full_code[task_key]['queries'][question_key][idx]['journal_score'],
        #                    full_code[task_key]['queries'][question_key][idx]['paper_citations_count'],
        #                    ]
        #             results_table.loc[cnt] = row
        #             cnt += 1
        # return [results_table]



def layoutStyle():
    style = """
        .single_answer {
            border-left: 3px solid red;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;
        }
        .answer{
            color: red;
        }    
    """
    return "<style>" + style + "</style>"

def getAnswerDiv(i, answ_id, answer, button=True):
    div = """<div id="{answ_id}" class="tab-pane fade">"""
    if i is 0:
        div = """<div id="{answ_id}" class="tab-pane fade in active">"""
    div += """
        <h2>Answer</h2>
        <p>{answer}</p>
        <h3>Title</h3>
        <p>{title}</p>
        <h3>Author(s)</h3>
        <p>{author}</p>
        <h3>Journal</h3>
        <p>{journal}</p>
        """
    if button:
        div += getAnswerAddInfo("ad_{}".format(answ_id), answer)
    div += """</div>"""
    return div.format(
        answ_id=answ_id, 
        answer=answer['answer'], 
        title=answer['title'],
        author=answer['author'], 
        journal=answer['journal'])

def getAnswerAddInfo(answ_id, answer):
    div = """
        <button type="button" class="btn-warning" data-toggle="collapse" data-target="#{answ_id}">Additional Info</button>
        <div id="{answ_id}" class="collapse">
        <h4>Scimago Journal Score</h4>
        <p>{journal_score}</p>
        <h4>Paper citations</h4>
        <p>{paper_citations_count}</p>
        </div>
        """
    return div.format(
        answ_id=answ_id, 
        journal_score=answer['journal_score'], 
        paper_citations_count=answer['paper_citations_count'])

def getAnswerLi(i, answer_id):
    div = """<li><a data-toggle="tab" href="#{ansid}">Ans {i}</a></li>"""
    if i is 0:
        div = """<li class="active"><a data-toggle="tab" href="#{ansid}">Ans {i}</a></li>"""
    return div.format(i=i+1, ansid=answer_id)

def getQuestionDiv(question, answers, topic_id, question_id, button=True):
    div = """
    <div>
      <button type="button" class="btn btn-info" data-toggle="collapse" data-target="#{question_id}">{question}</button>
      <div id="{question_id}" class="collapse">
        <nav class="navbar navbar-light" style="background-color: #E3F2FD; width: 400px">
            <div>
                <ul class="nav navbar-nav nav-tabs">""".format(
        question_id="{}_{}".format(topic_id, question_id), question=question)
    for i, answer in enumerate(answers):
        div += getAnswerLi(i, "{}_{}_{}".format(topic_id, question_id, i+1))
    div += """</ul>
        </div>
    </nav>
    <div class="tab-content">"""
    for i, answer in enumerate(answers):
        div += getAnswerDiv(
            i, "{}_{}_{}".format(topic_id, question_id, i+1), answer)
    return div + """</div> <br> </div> </div>"""

def getTaskDiv(task, questions):
    """
    :param task:
    :param questions: dict
    {question: [{"answer": "", "title": "", "journal": "", "authors": "", "journal_score": "", 
     "paper_citations_count": ""}]
    :return:
    """
    topic_id = task.replace(" ", "").replace("?", "")
    quest_id = 0
    questions_div = ""
    queries = questions['queries']
    for question, answers in queries.items():
        questions_div += getQuestionDiv(question, answers, topic_id, quest_id)
        quest_id += 1
    topic_header = """
       <div>
         <button type="button" class="btn" data-toggle="collapse" data-target="#{id1}" style="font-size:20px">&#8226{task}: {task_name}</button>
         <div id="{id1}" class="collapse">
         {body}
         </div>
       </div>""".format(id1=topic_id, task=task, body=questions_div, task_name=questions['task_name'])
    return topic_header

def getHtmlCode(tasks, style):
    header = """
        <!DOCTYPE html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
          <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
          {style}
        </head>
        <body> """.format(style=style)
    body = ""
    for task, questions in tasks.items():
        body += getTaskDiv(task, questions)
    html_code = header + body
    html_code += "</body>"
    return html_code

class QuestionAnswering(BaseModel):
    """
    Implementation of QuestionAnswering model
    """
    @staticmethod
    def get_knob_config():
        return {
            'to_eval': FixedKnob(False),
            'model_class':CategoricalKnob(['question_answering']),
        }
    def __init__(self, **knobs):
        super().__init__(**knobs)
        self._num_papers = 2

    def _create_model(self, scratch: bool):
        # if scratch == False:
        qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD);
        qa_model = qa_model.to(torch_device)
        qa_model.eval()
        return qa_model
    

    def train(self, dataset_path=None, shared_params: Optional[Params] = None, **train_args):
        self._model = self._create_model(scratch = False)

    def evaluate(self, dataset_path):
        return float(1)

    def predict(self, queries): 
        return QA(queries, self._model, self._num_papers)

    def dump_parameters(self):
        params = {
        }

        return params

    def load_parameters(self, params):
        self._model = self._create_model(scratch=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_file_path', type=str, default= "examples/data/question_answering/SampleQuestions.json", help='txt file which contains questions')
    (args, _) = parser.parse_known_args()

    queries = open(args.queries_file_path,'r')
    queries = queries.read().replace("'", "\"")
    queries = json.loads(queries)
    queries = [queries]


    test_model_class(
        model_file_path=__file__,
        model_class='QuestionAnswering',
        task='question_answering', 
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            'semanticscholar': '0.1.4',
            'sentence_transformers': '0.2.6.1'
        },
        train_dataset_path='',
        val_dataset_path='',
        queries=queries
    )

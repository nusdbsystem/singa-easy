from __future__ import division
from __future__ import print_function
import argparse
import glob
# from typing import Union, Dict, Optional, Any
from typing import Any, List

from singa_auto.model import CategoricalKnob, FixedKnob, utils, BaseModel
# from singa_auto.model.knob import BaseKnob

# PyTorch Dependency
import torch

# Misc Third-party Dependency
import numpy as np
import pandas as pd
import json
import re
import base64
import tempfile
import zipfile

import semanticscholar as sch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

BERT_SQUAD = "bert-large-uncased-whole-word-masking-finetuned-squad"
PRETRAINED_SENTENCETRANSFORMER = "distilbert-base-nli-stsb-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(BERT_SQUAD)
puncts = ['!', '.', '?', ';']
puncts_tokens = [tokenizer.tokenize(x)[0] for x in puncts]

class JsonFileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            try:
                self.paper_id = content['paper_id']
                self.title = content['metadata']['title']
            except:
                self.paper_id = []
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

class DataLoader:
    @classmethod
    def json_load_to_df(cls, all_json, metadata_path, topic_words=[], topic_year=2019):
        try: 
            meta_df = pd.read_csv(
                metadata_path,
                dtype={
                    'pubmed_id': str,
                    'Microsoft Academic Paper ID': str,
                    'doi': str
                }
            )
        except:
            pass

        dict_ = {
            'paper_id': [],
                 'abstract': [],
                 'body_text': [],
                 'title': [], 
                 'publish_time': [],
                 'abstract_summary': []
                 }
        for idx, entry in enumerate(all_json):
            if idx % (len(all_json) // 10) == 0:
                print(f'Processing index: {idx} of {len(all_json)}')
            content = JsonFileReader(entry)
            # get metadata information
            try:
                meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
                # add the title information
                dict_['title'].append(meta_data['title'].values[0])
                # add the publishing data
                dict_['publish_time'].append(meta_data['publish_time'].values[0])
            except:
                # add the title information
                dict_['title'].append(content.title)
                # add the publishing data
                dict_['publish_time'].append('')
            try:
                dict_['paper_id'].append(content.paper_id)
                dict_['abstract'].append(content.abstract)
            except:
                pass

            dict_['body_text'].append(content.body_text)
            df_topic = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'title', 'publish_time'])
            # removal of repeated data and duplicates
            df_topic.drop_duplicates(['title'], inplace=True)
            df_topic.dropna(subset=['body_text'], inplace=True)

            # Some papers in the dataset concern viruses different from SARS-CoV-2 (close parents such as SARS or MERS).
            # Here, we identify papers which specifically talk about the 2019 coronavirus,
            # by simply checking the publishing date of the paper and if some keywords occur in the body text.
            # Moreover, we remove papers published before 2019.
            if topic_words == []:
                df_topic_only = df_topic
            else:
                topic_words = [elem.lower() for elem in topic_words]
                topic_words = re.compile('|'.join(topic_words))
                df_topic['is_topic'] = df_topic.apply(cls.__checkTopic, axis=1, topic_words=topic_words, topic_year=topic_year)

                # We extracted the dataframe of papers concerning the SARS-CoV-2 virus
                df_topic_only = df_topic[df_topic['is_topic'] == True]
                df_topic_only = df_topic_only.reset_index(drop=True)

            df_topic_only['preproc_body_text'] = df_topic_only['body_text'].apply(cls.__preprocessing)

            # We proceed splitting all papers into chunks, each of which containing approximately 90 tokens.

            df_topic_only['body_text_parags'] = df_topic_only['preproc_body_text'].apply(cls.__splitParagraph)

            return df_topic_only

    @classmethod
    def load_to_embedding(cls, df_topic_only):

        assert df_topic_only is not None

        text = df_topic_only['body_text_parags'].to_frame()
        body_texts = text.stack().tolist()

        # Here, we instantiate the Sentence-BERT model that will be used to embed the chunks. We choose the lighter
        # architecture 'distilbert', in order to obtain embedding vectors in a reasonable time

        encoding_model = SentenceTransformer(PRETRAINED_SENTENCETRANSFORMER)

        # proceed embedding all the chunks of all the papers.
        topic_encoded = []
        for body in tqdm(body_texts):
            topic_encoded.append(encoding_model.encode(body, show_progress_bar=False))

        return topic_encoded, encoding_model

    @staticmethod
    def __checkYear(date):
            return int(date[0:4])

    @classmethod
    def __checkTopic(cls, row, topic_words, topic_year):
        if row['publish_time'] is not '' and topic_year is not None:
            if len(topic_words) == 0:
                return cls.__checkYear(row['publish_time']) > topic_year
            else:
                return bool(topic_words.search(row['body_text'].lower())) and cls.__checkYear(row['publish_time']) > topic_year
        else:
            return bool(topic_words.search(row['body_text'].lower()))


    @staticmethod
    def __preprocessing(text):
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
        text = re.sub(r'\[[\d\,\s]+\]', ' ', text)
        # replace tags (e.g. (NUM1) repeated at least 3 times with whitespace
        text = re.sub(r'(\(\d+\)\s){3,}', ' ', text)
        # replace '1.3' with '1,3' (we need it for split later)
        text = re.sub(r'(\d+)\.(\d+)', '\g<1>,\g<2>', text)
        # remove all full stops as abbreviations (e.g. i.e. cit. and so on)
        text = re.sub(r'\.(\s)?([^A-Z\s])', ' \g<1>\g<2>', text)
        # correctly spacing the tokens
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        # return lowercase text
        return text.lower()

    @staticmethod
    def __checkAnyStop(token_list, token_stops):
        return any([stop in token_list for stop in token_stops])

    @staticmethod
    def __firstFullStopIdx(token_list, token_stops):
        """
        Returns the index of first full-stop token appearing.
        """
        idxs = []
        for stop in token_stops:
            if stop in token_list:
                idxs.append(token_list.index(stop))
        minIdx = min(idxs) if idxs else None
        return minIdx

    @classmethod
    def __splitTokens(cls, tokens, puncts_tokens, split_length):
        """
        To avoid splitting a sentence and lose the semantic meaning of it, a paper is splitted
        into chunks in such a way that each chunk ends with a full-stop token (['.' ';' '?' or '!'])
        """

        splitted_tokens = []
        while len(tokens) > 0:
            if len(tokens) < split_length or not cls.__checkAnyStop(tokens, puncts_tokens):
                splitted_tokens.append(tokens)
                break
            # to not have too long parapraphs, the nearest fullstop is searched both in the previous
            # and the next strings.
            prev_stop_idx = cls.__firstFullStopIdx(tokens[:split_length][::-1], puncts_tokens)
            next_stop_idx = cls.__firstFullStopIdx(tokens[split_length:], puncts_tokens)
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

    @classmethod
    def __splitParagraph(cls, text, split_length=90):
        tokens = tokenizer.tokenize(text)
        splitted_tokens = cls.__splitTokens(tokens, puncts_tokens, split_length)
        return [tokenizer.convert_tokens_to_string(x) for x in splitted_tokens]

class QuestionAnswering(BaseModel):
    """
    Implementation of QuestionAnswering model
    """
    @staticmethod
    def get_knob_config():
        return {
            'to_eval': FixedKnob(False),
        }
    
    def __init__(self, **knobs):
        super().__init__()
        self._num_papers = 5
        self.topic_year = None
        self.topic_words = []
        # self.topic_year = 2019
        # self.topic_words = ['covid', 'coronavirus disease 19', 'sars cov 2', '2019 ncov', '2019ncov', '2019 n cov',
                           # '2019n cov','ncov 2019', 'n cov 2019', 'coronavirus 2019', 'wuhan pneumonia', 'wuhan virus',
                           # 'wuhan coronavirus', 'coronavirus 2', 'covid-19', 'SARS-CoV-2', '2019-nCov']
        self.df_topic_only = None
        self.topic_encoded = None
        self.encoding_model = None
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.qa_tokenizer = tokenizer 

    def _create_model(self, scratch: bool):
        qa_model = AutoModelForQuestionAnswering.from_pretrained(BERT_SQUAD);
        qa_model = qa_model.to(self.torch_device)
        qa_model.eval()
        return qa_model

    def train(self, dataset_path=None, shared_params=None, **train_args):
        pass

    def evaluate(self, dataset_path):
        return float(1)


    def dump_parameters(self):
        params = {}
        return params

    def load_parameters(self, params):
        # if zip file is passed as params in dict format
        if type(params) == dict:
            params = params['zip_file_base64']
            params = base64.b64decode(params.encode('utf-8'))
        # if file path is passed as params in str format
        elif type(params) == str: 
            # if finetune_dataset_path is send through params
            with open(params, 'rb') as f:
                params = f.read()

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            with open(tmp.name, 'wb') as f:
                f.write(params)
            with tempfile.TemporaryDirectory() as root_path:
                dataset_zipfile = zipfile.ZipFile(tmp.name, 'r')
                dataset_zipfile.extractall(path=root_path)
                dataset_zipfile.close()

                metadata_path = "{}/metadata.csv".format(root_path)
                all_json = glob.glob("{}/**/*.json".format(root_path), recursive=True)

                self.df_topic_only = DataLoader.json_load_to_df(
                    all_json=all_json,
                    metadata_path=metadata_path, topic_words=self.topic_words, topic_year=self.topic_year)

                self.topic_encoded, self.encoding_model = DataLoader.load_to_embedding(self.df_topic_only)

        self._model = self._create_model(scratch=False)

    def predict(self, queries: List[Any]) -> List[Any]:
        predictions = list()
        data = queries[0]
        assert isinstance(data, dict)

        try:
            task_questions = data['questions']
        except:
            task_questions = []
            for question_set in data.values():
                task_questions.extend(question_set['questions'])

        dict_queries = []
        for idx, query in enumerate(task_questions):
            print(f"Getting answers from query: {query}")
            answers_list = []
            if '\n' in query:
                '''this is for queries with /n separated context'''
                question, query_paragraph = query.splitlines()
                answer, conf, paragraph = self.__answerQuestion(question, query_paragraph, self._model)  # best_answer, confidence, best_paragraph
                if answer:
                    start, end = self.__findStartEndIndexSubstring(paragraph, answer)
                    answer_parag = paragraph[start:end]
                else:
                    answer_parag = 'idk'
                paper_answer = {"answer": answer_parag,}     
                answers_list.append(paper_answer)
                query = question
            else:
                '''this is for queries without context'''
                indexes_papers = self.__extractPapersIndexes(query, num_papers=self._num_papers)
                (question, indexes_papers) = (query, indexes_papers)
                
                for paper_index in indexes_papers:
                    answer, conf, paragraph = self.__answerQuestion(question, self.df_topic_only['body_text_parags'][
                        paper_index], self._model)
                    if answer:
                        start, end = self.__findStartEndIndexSubstring(paragraph, answer)
                        answer_parag = paragraph[start:end]

                        ''' please do not change the return format, since this part is to comply with dev.py'''
                        answers_list.append(answer_parag)
                    else:
                        answer_parag = 'idk'
                    dict_queries.extend(answers_list)
        return [dict_queries]
    def __answerQuestion(self, question, paper, qa_model):
        """
        This funtion provides the best answer found by the Q&A model, the chunk containing it
        among all chunks of the input paper and the score obtained by the answer
        """
        inputs = [self.qa_tokenizer.encode_plus(
            question, paragraph, add_special_tokens=True, return_tensors="pt") for paragraph in paper]
        answers = []
        confidence_scores = []
        for n, Input in enumerate(inputs):
            input_ids = Input['input_ids'].to(self.torch_device)
            token_type_ids = Input['token_type_ids'].to(self.torch_device)
            if len(input_ids[0]) > 510:
                input_ids = input_ids[:, :510]
                token_type_ids = token_type_ids[:, :510]
            text_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids[0])
            start_scores, end_scores = qa_model(input_ids, token_type_ids=token_type_ids)
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores)
            # if the start token of the answer is contained in the question, the start token is moved to
            # the first one of the chunk
            check = text_tokens.index("[SEP]")
            if int(answer_start) <= check:
                answer_start = check + 1
            answer = self.qa_tokenizer.convert_tokens_to_string(text_tokens[answer_start:(answer_end + 1)])
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


    def __extractPapersIndexes(self, query, num_papers=5):

        """
        # compute the cosine similarity between the query embedding and all the chunks embeddings,
        # storing the obtained maximum cosine similarity for each paper. We then identify the most semantically-related
        # papers with respect to the query as those obtaining the 5 biggest cosine similarities.
        """

        def computeMaxCosine(encoded_query, encodings):
            cosines = cosine_similarity(encoded_query[0].reshape(1, -1), encodings)
            return float(np.ndarray.max(cosines, axis=1))
        encoded_query = self.encoding_model.encode([query.replace('?', '')], show_progress_bar=False)
        cosines_max = []
        for idx in range(len(self.topic_encoded)):
            paper = np.array(self.topic_encoded[idx])
            result = computeMaxCosine(encoded_query, paper)
            cosines_max.append(result)

        indexes_max_papers = np.array(cosines_max).argsort()[-num_papers:][::-1]
        return indexes_max_papers


    @staticmethod
    def __postprocessing(text):
        # capitalize the text
        text = text.capitalize()
        # '2 , 3' -> '2,3'
        text = re.sub(r'(\d) \, (\d)', "\g<1>,\g<2>", text)
        # full stop
        text = re.sub(r' (\.|\!|\?) (\w)', lambda pat: pat.groups()[0] + " " + pat.groups()[1].upper(), text)
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

    @staticmethod
    def __findStartEndIndexSubstring(context, answer):
        """
        Search of the answer inside the paragraph. It returns the start and end index.
        """
        search_re = re.search(re.escape(answer.lower()), context.lower())
        if search_re:
            return search_re.start(), search_re.end()
        else:
            return 0, len(context)


if __name__ == '__main__':
    from singa_auto.constants import ModelDependency
    from singa_auto.model.dev import test_model_class
    parser = argparse.ArgumentParser()
    # parser.add_argument('--queries_file_path', type=str, default= "examples/data/question_answering/SampleQuestions.json", help='txt file which contains questions')
    parser.add_argument('--fine_tune_dataset_path', type=str, default= "data/covid19data.zip", help='e.g. path of covid19data.zip') 
    (args, _) = parser.parse_known_args()

    # queries = open(args.queries_file_path,'r')
    # queries = queries.read().replace("'", "\"")
    # queries = json.loads(queries)
    # queries = [queries]
    queries = [{
          'questions': [
                        'How long individuals are contagious?',
                        # 'What is the range of the incubation period in humans?',
                        ]
          }]

    test_model_class(
        model_file_path=__file__,
        model_class='QuestionAnswering',
        task='question_answering_covid19', 
        dependencies={ 
            ModelDependency.TORCH: '1.3.1',
            "torchvision": "0.4.2",
            'semanticscholar': '0.1.4',
            'sentence_transformers': '0.2.6.1',
            "tqdm": "4.27",
        },
        train_dataset_path='',
        val_dataset_path='',
        fine_tune_dataset_path=args.fine_tune_dataset_path,
        queries=queries
    )

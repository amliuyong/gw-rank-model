from __future__ import print_function

import os
import json
import numpy as np
import flask
import tarfile
import glob
from tensorflow.contrib import predictor
import logging

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
info_file_prefix = '/opt/ml/info'


class Rank():
    def __init__(self):
        self.entity_embed = np.load(os.path.join(info_file_prefix, "dkn_entity_embedding.npy"))
        print("loaded dkn_entity_embedding.npy")
        self.context_embed = np.load(os.path.join(info_file_prefix, "dkn_context_embedding.npy"))
        print("loaded dkn_context_embedding.npy")
        self.word_embed = np.load(os.path.join(info_file_prefix, "dkn_word_embedding.npy"))
        print("loaded dkn_word_embedding.npy")

        # with open(os.path.join(info_file_prefix, "news_id_news_feature_dict.pickle"), "rb") as file_to_load:
        #      self.news_id_news_property = pickle.load(file_to_load)
        #      print("loaded news_id_news_feature_dict.pickle")

        tar = tarfile.open(os.path.join(info_file_prefix, "model.tar.gz"), "r")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, info_file_prefix)
        tar.close

        for name in glob.glob(os.path.join(info_file_prefix, '**', 'saved_model.pb'), recursive=True):
            print("found model saved_model.pb in {} !".format(name))
            model_path = '/'.join(name.split('/')[0:-1])
        self.model = predictor.from_saved_model(model_path)
        print("loaded saved_model.pb")
        print("Rank __init__ complete")

    def predict(self, data):
        '''
        data = {
             "news_words": [[21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
             "news_entities": [[2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
             "click_words": [
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [21, 327, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
             ],
             "click_entities": [
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2814, 2814, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
             ]
       }
        '''
        print('generate_rank_result start')

        news_words_index_np = np.array(data['news_words'])
        news_entity_index_np = np.array(data['news_entities'])
        click_words_index_np = np.array(data['click_words'])
        click_entity_index_np = np.array(data['click_entities'])

        print('start create input_dict')
        input_dict = {}
        input_dict['click_entities'] = self.entity_embed[click_entity_index_np]
        input_dict['click_words'] = self.word_embed[click_words_index_np]
        input_dict['news_entities'] = self.entity_embed[news_entity_index_np]
        input_dict['news_words'] = self.word_embed[news_words_index_np]
        print("check input shape!")

        print("input click entities shape {}".format(
            input_dict['click_entities'].shape))
        print("input click words shape {}".format(
            input_dict['click_words'].shape))
        print("input news entities shape {}".format(
            input_dict['news_entities'].shape))
        print("input news words shape {}".format(
            input_dict['news_words'].shape))

        output = self.model(input_dict)

        print('output {} from model'.format(output))

        output_prob = output['prob']
        rank_result = []
        i = 0
        while i < len(output_prob):
            rank_result.append(str(output_prob[i]))
            i = i + 1

        return rank_result


# The flask app for serving predictions
app = flask.Flask(__name__)

rank = Rank()


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here

    # status = 200 if health else 404
    status = 200
    return flask.Response(response='OK\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        print("raw data is {}".format(flask.request.data))
        data = flask.request.data.decode('utf-8')
        print("data is {}".format(data))
        data_input = json.loads(data)
        data = data_input['instances']
        print("final data is {}".format(data))
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # Do the prediction
    predictions = []
    input_len = len(data)

    if input_len == 0:
        raise "data_input['instances'] is empty"

    data_for_model = prepare_data_for_model(data)
    predictions.append(rank.predict(data_for_model)[0:input_len])
    print(predictions)
    print(json.dumps(np.asarray(predictions).tolist()))

    rr = json.dumps({'result': np.asarray(predictions).tolist()})
    print("bytes prediction is {}".format(rr))

    return flask.Response(response=rr, status=200, mimetype='application/json')


def prepare_data_for_model(data_arr):
    if len(data_arr) == 1:
        data_arr.append(data_arr[0])

    news_words = []
    news_entities = []
    click_words = []
    click_entities = []

    for inst in data_arr:
        news_words.extend(inst['news_words'])
        news_entities.extend(inst['news_entities'])
        click_words.extend(inst['click_words'])
        click_entities.extend(inst['click_entities'])

    ret_data = {
        "news_words": news_words,
        "news_entities": news_entities,
        "click_words": click_words,
        "click_entities": click_entities
    }

    print("prepare_data_for_model return {}".format(ret_data))
    return ret_data

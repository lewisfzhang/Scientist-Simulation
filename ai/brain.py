# brain.py
# the decision making process based on trained neural net and optimized with approximate DP
# acts as an artificial server

import subprocess as s
from tensorflow import keras
import numpy as np
import pickle, os


class Brain:
    # load the neural network into the brain
    def __init__(self, model):
        self.model = model

    def process(self, data):
        results = self.model.predict(data).flatten()
        # NOTE: everything is duplicated to simulate with and without funding (very important factor!)
        idea_choice = np.argmax(results)
        # format: idea_choice, exp_return, with_funding
        # disregard || no past funding --> 0 --> need funding
        # 1 = using funding, 0 = don't use funding
        return int(idea_choice/2), results[idea_choice], idea_choice % 2 == 1

    @staticmethod
    def load_brain():
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        loc = path + '/ai/results/model.h5'
        return Brain(keras.models.load_model(loc))

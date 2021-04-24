import argparse
import os
import utils
import torch


class UnionOptions(TrainOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        
    def initialize(self):
        TrainOptions.initialize(self)
        
        self.parser.add_argument('--model_type_1', type=str, default='lstm', help='model_type, lstm|cnn')
        self.parser.add_argument('--model_type_2', type=str, default='cnn_Res50_IR', help='model_type, lstm|cnn')        
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        return self.opt


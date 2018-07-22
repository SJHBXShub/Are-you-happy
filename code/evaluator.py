#! /usr/bin/python
# -*- coding: utf-8 -*-


import math


class Evaluator(object):

    valid_evaluator_name = ['entropy_loss']

    @staticmethod
    def evaluate(evaluator_name, labels, preds):
        assert evaluator_name.lower() in Evaluator.valid_evaluator_name, 'Wrong evaluator_name(%s)' % evaluator_name
        evaluator = getattr(Evaluator, evaluator_name)
        return evaluator(labels, preds)

    @staticmethod
    def entropy_loss(labels, preds):
        epsilon = 1e-15
        s = 0.
        for idx, l in enumerate(labels):
            assert l == 1 or l == 0
            score = preds[idx]
            score = max(epsilon, score)
            score = min(1 - epsilon, score)
            s += - l * math.log(score) - (1. - l) * math.log(1 - score)
        if len(labels) > 0:
            s /= len(labels)
        else:
            s = -1
        return s

    @staticmethod
    def accuracy(labels,preds,threshold=0.5):
        total_num = len(labels)
        correct_num = 0
        for idx, l in enumerate(labels):
            assert l == 1 or l == 0
            score = preds[idx]
            if score > threshold:
                score = int(1)
            else:
                score = int(0)
            if score == int(l):
                correct_num += 1
        return 1.0 * correct_num / total_num

    @staticmethod
    def analysis_result(labels,preds):
        epsilon = 1e-15
        posative_num = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        negative_num = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for idx, l in enumerate(labels):
            score = preds[idx]
            score = max(epsilon, score)
            score = min(1 - epsilon, score)
            class_lab = int(score * 10)
            if l == 1:
                posative_num[class_lab] += 1
            else:
                negative_num[class_lab] += 1
        print(posative_num)
        print(negative_num)




    @staticmethod
    def mean_value(x):
        return sum(x) * 1.0 / len(x)

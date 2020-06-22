#!/usr/bin/env python
import os
import unittest
from os.path import dirname

import DeepCas9.DeepCas9_TestCode
from DeepCas9.test import test_utils

root_dir = dirname(dirname(__file__))


class TestDeepCas9(unittest.TestCase):

    def test_end_to_end(self):
        test_out_dir = '{}/test_outputs'.format(root_dir)
        actual_output = '{}/test_outputs/predictions_DeepCas9_Final.txt'.format(root_dir)
        expected_output = '{}/test/resources/expected.training_sequences_head_DeepCas9_Final_predictions.txt'.format(
                                            root_dir)

        if os.path.exists(actual_output):
            os.remove(actual_output)
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        DeepCas9.DeepCas9_TestCode.process(['{}/test/resources/training_sequences_head.txt'.format(root_dir)],
                                           ['{}/DeepCas9_Final'.format(root_dir)],
                                   '{}/test_outputs/predictions'.format(root_dir))

        test_utils.assert_file_content_equals(self, actual_output, expected_output, ignore_pattern="^#")

#!/usr/bin/env python
import gzip
import os
import unittest
from os.path import dirname

import DeepCas9.deep_cas9

root_dir = dirname(dirname(__file__))


class TestDeepCas9(unittest.TestCase):

    def test_end_to_end(self):
        test_out_dir = '{}/test_outputs'.format(root_dir)
        actual_output = '{}/test_outputs/predictions_DeepCas9_Final.txt'.format(root_dir)

        os.remove(actual_output)
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        DeepCas9.deep_cas9.main(['./resources/training_sequences_head.txt'.format(root_dir),
                                 '{}/DeepCas9_Final'.format(root_dir),
                                 '{}/test_outputs/predictions'.format(root_dir)])

        self.assert_file_content_equals( '{}/test_outputs/predictions_DeepCas9_Final.txt'.format(root_dir),
                                         '{}/test/resources/expected.training_sequences_head_DeepCas9_Final_predictions.txt'.format(root_dir)
                                         )

    def assert_file_content_equals(self, actual_file, expected_file):

        with open(actual_file) as actual:
            with open(expected_file) as expected:
                self.assertListEqual(list(actual),list(expected))

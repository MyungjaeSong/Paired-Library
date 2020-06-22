import re


def assert_file_content_equals(test_class, actual_file, expected_file, ignore_pattern=None, ignore_fields=None, compare_sorted=False):
    if ignore_fields is not None:
        if not isinstance(ignore_fields, list):
            ignore_fields = [ignore_fields]
        ignore_fields.sort()
        ignore_fields.reverse()
    with open(actual_file) as actual:
        with open(expected_file) as expected:
            actual_list = list(actual)
            expected_list = list(expected)
            if compare_sorted:
                actual_list = sorted(actual_list)
                expected_list = sorted(expected_list)
            if ignore_pattern:
                actual_list = [s for s in actual_list if not re.search(ignore_pattern, s)]
                expected_list = [s for s in expected_list if not re.search(ignore_pattern, s)]
            if ignore_fields:
                if len(actual_list) != len(expected_list):
                    test_class.assertListEqual(actual_list, expected_list)
                else:
                    for i in range(len(actual_list)):
                        actual = actual_list[i].split()
                        expected = expected_list[i].split()
                        for j in ignore_fields:
                            if j < len(actual):
                                actual = actual[:j] + actual[j + 1:]
                            if j < len(expected):
                                expected = expected[:j] + expected[j + 1:]
                        test_class.assertListEqual(actual, expected)
            else:
                test_class.assertListEqual(actual_list, expected_list)

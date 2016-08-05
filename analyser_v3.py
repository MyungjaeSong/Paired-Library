#!/usr/bin/python
#-*- coding: utf-8 -*-
__author__ = 'forestkeep21@naver.com'

import re
import sys
import os

from Levenshtein import editops
# 해당 라이브러리 도큐먼트
# https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html

BASE_DIR = os.path.dirname(sys.executable)

#for debug
# BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def seq_validator(data):
    # 문자열이 시퀀스인지 판별. 판별 끝난 후 비교를 쉽게 하기 위해 모두 대문자로 변환한다.
    m = re.findall(r'^[A|a|T|t|C|c|G|g]+$', data)
    return m[0].upper() if m else None


def count_line_in_file(file_name):
    # 파일내 라인 카운터
    count = 0
    for line in open(file_name, 'r'):
        count += 1
    return count


def do(input_file_name, backward_target_length, dest_folder_path):
    cur_cnt = 0
    target_cnt = count_line_in_file(input_file_name)
    final_results = {}

    for target_set in open(input_file_name, 'r'):
        # 인풋 파일은 이름 : 와일드시퀀스 : 타겟 의 형태를 띈다.
        tmp = target_set.split(':')
        file_name_no_ext = tmp[0].strip()
        file_name = '{}.txt'.format(file_name_no_ext)
        wild_seq = tmp[1].strip()
        target = tmp[2].strip()

        # 타겟 검사
        target = seq_validator(target)
        if not target:
            continue

        # 와일드 시퀀스 검사
        wild_seq = seq_validator(wild_seq)
        if not wild_seq:
            continue

        # 결과 저장용 폴더 생성
        result_folder_name = os.path.join(BASE_DIR, 'analyse_results')
        if not os.path.exists(result_folder_name):
                os.makedirs(result_folder_name)

        try:
            # 결과 임시 저장 dict
            result = {
                'total_cnt': count_line_in_file(os.path.join(dest_folder_path, file_name)),
                'mutated_cnt': 0,
                'mutated_rates': 0.0,
                'mutated_dict': {}
            }

            for line in open(os.path.join(dest_folder_path, file_name), 'r'):
                # 대상 시퀀스 검사
                line = seq_validator(line)
                if not line:
                    continue

                # 와일드 시퀀스와 타겟을 이용하여 와일드 시퀀스에서 타겟의 시작, 종료 위치를 파악한다. editops에서 사용.
                target_start_pos_in_wild = int(wild_seq.find(target))
                target_end_pos = target_start_pos_in_wild + len(target)

                # 와일드 시퀀스를 기준으로 대상 시퀀스와 비교하여 레벤슈타인 유사도 측정에서 editops를 뽑아낸다.
                # editops는 (변형방법, 와일드시퀀스 기준 위치, 대상시퀀스 기준 위치) 의 형태로 결과가 나온다.
                # 예를 들어, editops('test', 'teaasz') 의 경우 [('insert', 2, 2), ('insert', 2, 3), ('replace', 3, 5)]
                # 1번 인덱스 : 삽입이 와일드시퀀스 기준 2번째, 대상시퀀스 기준 2번째에서 발생
                # 2번 인덱스 : 삽입이 와일드시퀀스 기준 2번째, 대상시퀀스 기준 3번째에서 발생
                # 3번 인덱스 : 교체가 와일드시퀀스 기준 3번째, 대상시퀀스 기준 5번째에서 발생
                # 때문에 와일드시퀀스에서 타겟의 위치만 정확히 파악한다면 대상시퀀스에서 변형이 어느부분에 일어났는지
                # 몰라도 사용자가 지정한 위치에서의 변형 여부를 충분히 잡아낼 수 있다.
                for mutation_info in editops(wild_seq, line):
                    # 사용자 지정 위치 검사(타겟의 뒤에서부터 backward_target_length 번째까지)
                    if target_end_pos - int(backward_target_length) <= mutation_info[1] <= target_end_pos:
                        # 교체는 변형으로 치지 않는다.
                        if mutation_info[0] != 'replace':
                            # 여기까지 왔다면 변형으로 쳐서 카운트+1
                            result['mutated_cnt'] += 1
                            # 변형된 대상시퀀스를 결과 출력을 위해 저장하고 동일 시퀀스 갯수 조사를 위해 카운팅한다.
                            if line not in result['mutated_dict'].keys():
                                result['mutated_dict'][line] = 1
                            else:
                                result['mutated_dict'][line] += 1
                            break

            # 변형 퍼센티지 계산
            try:
                result['mutated_rates'] = float(result['mutated_cnt']) / result['total_cnt'] * 100
            except:
                result['mutated_rates'] = 0

            # 각 결과값 저장.
            with open(os.path.join(result_folder_name, file_name), 'w') as f:
                for mutated_seq, cnt in result['mutated_dict'].items():
                    f.write('{} X {}\n'.format(mutated_seq, cnt))
                f.write('--------\n')
                f.write('mutation rates : {} %'.format(result['mutated_rates']))

        except Exception as e:
            print e
            print file_name, ' not found.'
            pass
        else:
            # 문제 없다면 결과물을 모은다.
            final_results[file_name_no_ext] = result

        # 타겟 하나 분석 종료 카운트+1
        cur_cnt += 1
        # 진행율 화면 표시
        progress_percentage = float(cur_cnt) / target_cnt * 100
        print '{} % done'.format(progress_percentage)

        # 최종 결과물 파일 저장.
        with open(os.path.join(result_folder_name, 'result_info.txt'), 'w') as f:
            for name, data in final_results.items():
                f.write('{} : {} : {}/{}\n'.format(name, data['mutated_rates'], data['mutated_cnt'], data['total_cnt']))

if __name__ == '__main__':
    print u'Input file name with extension: '
    # 이름 : 와일드시퀀스 : 타겟 으로 구성된 파일을 입력받는다.
    input_file_name = raw_input()
    input_file_name = os.path.join(BASE_DIR, input_file_name)

    if not os.path.isfile(input_file_name):
        print u'File Not Found. Check it is in same folder'
        raise

    print u'Input length to check mutation from backward of target: '
    # 사용자 지정 위치를 입력받는다. 타겟의 제일 위에서부터 ~번째이다.
    backward_target_length = raw_input()

    print u'Input result folder name: '
    # 추출기가 뽑아낸 대상시퀀스들이 모여있는 폴더 이름 입력.
    dest_folder_name = raw_input()
    dest_folder_name = os.path.join(BASE_DIR, dest_folder_name)

    if not os.path.isdir(dest_folder_name):
        print u'Folder Not Found'
        raise

    # 분석시작
    do(input_file_name, backward_target_length, dest_folder_name)

    print u'Well done. Press any key'
    raw_input()


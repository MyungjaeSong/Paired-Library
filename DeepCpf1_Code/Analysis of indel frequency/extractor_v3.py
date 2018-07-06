#!/usr/bin/python
#-*- coding: utf-8 -*-
__author__ = 'forestkeep21@naver.com'

import re
import sys
import os

BASE_DIR = os.path.dirname(sys.executable)

#for debug
# BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def seq_validator(data):
    m = re.findall(r'^[A|a|T|t|C|c|G|g]+$', data)
    return m[0] if m else None


def count_line_in_file(file_name):
    count = 0
    for line in open(file_name, 'r'):
        count += 1
    return count


def do(src_file_name, dest_file_name):
    # 프로그램 진행율을 계산하기 위해 파일의 라인수를 센다.
    src_line_cnt = count_line_in_file(src_file_name)
    if src_line_cnt == 0:
        print u'File Not Found'
        raise
    current_cnt = 0
    extracted_line_index = []

    # 결과가 저장될 폴더 지정
    result_folder_name = os.path.join(BASE_DIR, 'results')
    # 추출할 시퀸스가 있는 파일을 읽어온다.
    data = [line.strip() for line in open(dest_file_name, 'r') if seq_validator(line)]
    # 바코드가 있는 파일을 읽어온다.
    barcode_data = [line for line in open(src_file_name, 'r')]

    # 결과가 저장될 폴더가 없다면 하나 생성
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)

    try:
        # 읽어온 바코드를 속도를 위해 모두 메모리에 올려놓고 분석을 시작한다.
        for barcode in barcode_data:
            # 바코드셋은 :를 구분자로 앞은 파일명, 뒤는 바코드로 되어있다.
            barcode_set = barcode.split(':')
            if len(barcode_set) < 2:
                continue
            # 파일명에서 화이트 스페이스 삭제
            file_name = barcode_set[0].strip()
            # 바코드가 valid한지 검증
            barcode = seq_validator(barcode_set[1].strip())

            used_data = []
            # 대상이 되는 시퀸스들을 하나하나 분석한다.
            for line in data:
                # 대상 시퀸스 valid 검증
                line = seq_validator(line)
                if line is None:
                    continue

                # 비교를 위해 바코드, 대상 시퀸스 둘다 소문자로 변환하여 바코드가 대상 시퀸스 내에 존재하는지 검사
                if barcode.lower() in line.lower():
                    # 존재한다면 대상 시퀸스는 이제 필요없으므로 추후 메모리에서 제거하기 위해 따로 보관한다.
                    used_data.append(line)

            # 결과가 저장될 파일명 지정
            file_name = os.path.join(result_folder_name, '{}.txt'.format(file_name))
            # 결과 파일 쓰기 시작
            with open(file_name, 'w') as f:
                # 추출된 대상 시퀸스들을 파일에 쓴다.
                for datum in used_data:
                    f.write('{}\n'.format(datum))

            # 파일에 전부 옮겨담았다면 메모리에 올라간 전체 대상 시퀸스들에서 파일에 쓴 대상 시퀸스를 뺀다.
            [data.remove(used_datum) for used_datum in used_data]

            # 프로그램 진행율 계산 부분
            current_cnt += 1
            progress_percentage = (float(current_cnt) / src_line_cnt * 100)
            print u'{} %'.format(progress_percentage)

    except Exception as e:
        print e
        print u'Extraction Failure.'
        raise

    try:
        # 모든 바코드의 분석이 종료되었다면 총 결과파일을 쓴다. 총 결과 파일명 지정
        result_info_file_name = os.path.join(result_folder_name, 'result_info.txt')
        with open(result_info_file_name, 'w') as f:
            # 각 개별 결과 파일을 열어서
            for line in open(src_file_name, 'r'):
                barcode_set = line.split(':')
                file_name = barcode_set[0].strip()
                # 라인 수를 센다음에
                count = count_line_in_file(os.path.join(BASE_DIR, result_folder_name, '{}.txt'.format(file_name)))
                # 총 결과 파일에 파일명 : 라인수 형식으로 쓴다.
                f.write('{} : {}\n'.format(file_name, count))
    except Exception as e:
        print e
        print u'Extraction has been done. But Making a result-info.txt is failed.'
        raise

if __name__ == "__main__":
    print u'Input barcode file name with extension: '
    src_file_name = raw_input()
    src_file_name = os.path.join(BASE_DIR, src_file_name)

    if not os.path.isfile(src_file_name):
        print u'File Not Found. Check it is in same folder'
        raise

    print u'Input sequence file name with extension: '
    dest_file_name = raw_input()
    dest_file_name = os.path.join(BASE_DIR, dest_file_name)

    if not os.path.isfile(dest_file_name):
        print u'File Not Found. Check it is in same folder'
        raise

    do(src_file_name, dest_file_name)

    print u'Well done. Press any key'
    raw_input()


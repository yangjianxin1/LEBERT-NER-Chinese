import json
from tqdm import tqdm, trange
import os
from os.path import join
from utils.utils import write_lines, load_lines
from loguru import logger
import os


def bmes_to_json(bmes_file, json_file):
    """
    将bmes格式的文件，转换为json文件，json文件包含text和label,并且转换为BIOS的标注格式
    Args:
        bmes_file:
        json_file:
    :return:
    """
    texts = []
    with open(bmes_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        words = []
        labels = []
        for idx in trange(len(lines)):
            line = lines[idx].strip()

            if not line:
                assert len(words) == len(labels), (len(words), len(labels))
                sample = {}
                sample['text'] = words
                sample['label'] = labels
                texts.append(json.dumps(sample, ensure_ascii=False))

                words = []
                labels = []
            else:
                word, label = line.split()
                label = label.replace('M-', 'I-').replace('E-', 'I-')
                words.append(word)
                labels.append(label)

    with open(json_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write("{}\n".format(text))


def bmes_to_json2(bmes_file, json_file):
    result = []
    lines = load_lines(bmes_file)
    for line in lines:
        line = line.strip()
        text, labels = line.split('\t')
        text = text.split()
        labels = [label.replace('M-', 'I-').replace('E-', 'I-') for label in labels.split()]
        assert len(text) == len(labels)
        sample = {'text': text, 'label': labels}
        result.append(json.dumps(sample, ensure_ascii=False))
    write_lines(result, json_file)


def get_label_tokens(input_file, output_file):
    """
    从数据集中获取所有label
    :return:
    """
    labels = set()
    lines = load_lines(input_file)
    for line in lines:
        data = json.loads(line)
        for label in data['label']:
            label = label.replace('M-', 'I-').replace('E-', 'I-')
            labels.add(label)
    labels = list(labels)
    labels = sorted(labels)
    labels.remove('O')
    labels.insert(0, 'O')
    logger.info('len of label:{}'.format(len(labels)))
    write_lines(labels, output_file)


def convert_cner():
    # bmes生成json
    bmes_files = ['../datasets/cner/dev.txt', '../datasets/cner/test.txt', '../datasets/cner/train.txt']
    for bmes_file in bmes_files:
        dirname = os.path.dirname(bmes_file)
        file_name = os.path.basename(bmes_file).split('.')[0] + '.json'
        json_file = join(dirname, file_name)
        bmes_to_json(bmes_file, json_file)

    # 生成label文件
    input_file = '../datasets/cner/train.txt'
    output_file = '../datasets/cner/labels.txt'
    get_label_tokens(input_file, output_file)


def convert_weibo():
    # bmes生成json
    bmes_files = ['../datasets/weibo/dev.char.bmes', '../datasets/weibo/test.char.bmes', '../datasets/weibo/train.char.bmes']
    for bmes_file in bmes_files:
        dirname = os.path.dirname(bmes_file)
        file_name = os.path.basename(bmes_file).split('.')[0] + '.json'
        json_file = join(dirname, file_name)
        bmes_to_json(bmes_file, json_file)

    # 生成label文件
    input_file = '../datasets/weibo/train.char.bmes'
    output_file = '../datasets/weibo/labels.txt'
    get_label_tokens(input_file, output_file)


if __name__ == '__main__':
    # 生成json文件
    # data_names = ['msra', 'ontonote4', 'resume', 'weibo']
    # path = '../datasets'
    # for data_name in data_names:
    #     logger.info('processing dataset:{}'.format(data_name))
    #     files = os.listdir(join(path, data_name))
    #     for file in files:
    #         file = join(path, data_name, file)
    #         data_type = os.path.basename(file).split('.')[0]
    #         out_path = join(path, data_name, data_type+'.json')
    #         if data_name in ['msra', 'weibo']:
    #             bmes_to_json(file, out_path)
    #         else:
    #             bmes_to_json2(file, out_path)


    # 生成label文件
    data_names = ['msra', 'ontonote4', 'resume', 'weibo']
    path = '../datasets'
    for data_name in data_names:
        logger.info('processing dataset:{}'.format(data_name))
        input_file = join(path, data_name, 'train.json')
        output_file = join(path, data_name, 'labels.txt')
        get_label_tokens(input_file, output_file)

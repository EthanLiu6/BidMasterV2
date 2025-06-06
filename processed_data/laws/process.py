"""
通过大模型把原始pdf、word等法律文档提取成json/csv格式：
{
    '条款': 'xxxxx',
    '内容': 'xxx'
}

然后转成csv/json（可能后面更好处理，暂时先保留csv/json）
"""

from utils import common_utils

project_path = common_utils.find_project_root()


def laws_json2csv(cur_json, target_csv):
    common_utils.json2csv(
        project_path.joinpath(cur_json),
        project_path.joinpath(target_csv)
    )


def laws_csv2json(cur_csv, target_json):
    common_utils.csv2json(
        project_path.joinpath(cur_csv),
        project_path.joinpath(target_json),
        encoding='utf-8-sig'
    )


if __name__ == '__main__':
    # laws_json2csv('processed_data/laws/招标投标领域公平竞争审查规则.json',
    #               'processed_data/laws/招标投标领域公平竞争审查规则.csv')

    # laws_csv2json('processed_data/laws/招标投标领域公平竞争审查规则.csv',
    #               'processed_data/laws/招标投标领域公平竞争审查规则.json')
    # laws_csv2json('processed_data/laws/政府采购质疑和投诉办法.csv',
    #               'processed_data/laws/政府采购质疑和投诉办法.json')
    # laws_csv2json('processed_data/laws/中华人民共和国招标投标法实施条例.csv',
    #               'processed_data/laws/中华人民共和国招标投标法实施条例.json')
    # laws_csv2json('processed_data/laws/中华人民共和国政府采购法.csv',
    #               'processed_data/laws/中华人民共和国政府采购法.json')
    # laws_csv2json('processed_data/laws/中华人民共和国政府采购法实施条例.csv',
    #               'processed_data/laws/中华人民共和国政府采购法实施条例.json')
    laws_csv2json('processed_data/laws/政府采购货物和服务招标投标管理办法.csv',
                  'processed_data/laws/政府采购货物和服务招标投标管理办法.json')

    laws_csv2json('processed_data/laws/政府采购促进中小企业发展管理办法.csv',
                  'processed_data/laws/政府采购促进中小企业发展管理办法.json')


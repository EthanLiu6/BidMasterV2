from utils import common_utils

project_path = common_utils.find_project_root()

if __name__ == '__main__':
    common_utils.excel2json(
        project_path.joinpath('processed_data/QA/QA_with_legal_basis.xlsx'),
        project_path.joinpath('processed_data/QA/QA_with_legal_basis.json')
    )
    # common_utils.excel2json(project_path.joinpath('raw_data/biaodewu/招标采购标的物信息提取训练数据.xlsx'),
    #                         project_path.joinpath('processed_data/biaodewu/biaodewu_info.json'),
    #                         origin_sheet_name=1)

    # common_utils.excel2json(project_path.joinpath('raw_data/biaodewu/问题答案样例数据0113.xlsx'),
    #                         project_path.joinpath('processed_data/QA/QA_demo.json')
    #                         )

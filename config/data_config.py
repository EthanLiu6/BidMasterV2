from config import ROOT_PATH

# 用于初始化存储时候的法律数据文件夹地址
LAWS_DIR_PATH = ROOT_PATH.joinpath('processed_data/laws')
# 用于给数据库增加法律数据的法律文件地址
LAW_ADD_FILE_PATH = None

# 用于初始化存储时候的标的物信息数据文件夹地址
BIAODEWU_DIR_PATH = ROOT_PATH.joinpath('processed_data/biaodewu')
# 用于给数据库增加标的物信息数据的文件地址
BIAODEWU_ADD_FILE_PATH = None

# 问题类别，不要随便改！！！！跟整体链路和知识库内容密切相关（数据管理部分人员整理）
# Note: 若进行修改，需要修改query_cls_and_select_branch，COLLECTIONS等内部代码
QUERY_CLS = ('招投标法律法规相关', '招投标公示信息相关', '与招投标无关问题')

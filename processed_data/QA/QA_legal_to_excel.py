"""
主要功能：从现有doc/docx的规范文档里面提取问答内容（原始数据已是半规范的）
"""

import os
import re
import pandas as pd
from docx import Document


def extract2excel_qa_law_from_text(text):
    qa_list = []
    # 将全文按字段切割，支持任意换行方式
    pattern = re.compile(
        r"问[:：](.*?)答[:：](.*?)法律依据为(.*?)(?=(问[:：])|$)",
        re.DOTALL
    )
    matches = pattern.findall(text)
    for q, a, law, *_ in matches:
        qa_list.append({
            "问": q.strip().replace('\n', ' '),
            "答": a.strip().replace('\n', ' '),
            "法律依据": law.strip().replace('\n', ' ')
        })
    return qa_list


def extract_from_docx(file_path):
    doc = Document(file_path)
    full_text = '\n'.join([para.text for para in doc.paragraphs])
    return extract2excel_qa_law_from_text(full_text)


def process_all_documents(folder_path):
    all_qa = []
    for file in os.listdir(folder_path):
        if file.endswith('.docx'):
            file_path = os.path.join(folder_path, file)
            qa_items = extract_from_docx(file_path)
            all_qa.extend(qa_items)
    return all_qa


if __name__ == "__main__":
    folder = "../../raw_data/QA/Bid_QA_dist_data01/"
    output_excel = "./QA_with_legal_basis.xlsx"

    qa_data = process_all_documents(folder)
    df = pd.DataFrame(qa_data)
    df.to_excel(output_excel, index=False)
    print(f"✅ 已提取 {len(df)} 条问答，保存至：{output_excel}")

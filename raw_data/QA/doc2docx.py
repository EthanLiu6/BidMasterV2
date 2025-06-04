import os
import subprocess


def doc2docx(_source_dir_path, _save_file_path):
    source = _source_dir_path
    dest = _save_file_path
    g = os.listdir(source)
    file_path = [f for f in g if f.endswith(('.doc'))]
    print(file_path)
    for i in file_path:
        file = (source + '/' + i)
        print(file)
        output = subprocess.check_output(
            ["/Applications/LibreOffice.app/Contents/MacOS/soffice", "--headless", "--convert-to", "docx", file,
             "--outdir",
             dest])
    print('success!')


if __name__ == '__main__':
    source_path = 'Bid_QA_source_data01'
    save_file_path = 'Bid_QA_dist_data01'
    doc2docx(_source_dir_path=source_path, _save_file_path=save_file_path)

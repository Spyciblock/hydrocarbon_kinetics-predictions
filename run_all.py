from pathlib import Path
from subprocess import run
import argparse
import json

for param_file in Path('experiments').glob('*.json'):
    print(f'Run {param_file.stem}')
    run(['python', 'train_model_test.py', str(param_file)])
#    parser = argparse.ArgumentParser()
#    parser.add_argument(param_file, type=Path)
#    args = parser.parse_args()
    param = json.load(param_file.open())
    print(param['name_training'])
    run(['cp', param_file, param['foldername_save'] + 'saved_model/' + param['name_training'] + '/'])
    run(['cp', './model_2.py', param['foldername_save'] + 'saved_model/' + param['name_training'] + '/'])

"""
    e.g., python submit.py job_config.yaml
"""
import argparse
import subprocess
from src import parse_job_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to job config file (.yaml)')
    args = parser.parse_args()

    submisson_cmd, _ = parse_job_config(args.config)
    msg = subprocess.check_output(submisson_cmd, shell=True)
    print(msg.decode('utf-8'))

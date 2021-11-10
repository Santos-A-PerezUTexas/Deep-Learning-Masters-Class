from .tournament import grade
from pathlib import Path
import logging
import json


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-w', '--workdir', default='/tmp/_workdir/')
parser.add_argument('-l', '--log', default='INFO')
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.log.upper()), format='%(message)s')

workdir = Path(args.workdir)
submissions = {f.name.replace('.zip', ''): f for f in workdir.glob('*.zip')}

grades = grade(submissions, workdir)

with open(workdir / 'grades.json', 'w') as f:
    json.dump(grades, f)

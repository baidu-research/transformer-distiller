import os
pwd=os.path.dirname(__file__)
METHODS ={
    'distilbert': os.path.join(pwd, 'distilbert/main.py'),
    'tinybert': os.path.join(pwd, 'tinybert/main.py'),
    'minilm2': os.path.join(pwd, 'minilm2/main.py')
}

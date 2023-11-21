import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    GITHUB_URL = os.environ.get('GITHUB_URL') or 'https://github.com/florianehmann/twitter-emotion'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'twitter-emotion-secret'

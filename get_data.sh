#!/usr/bin/env bash
wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
p7zip -d stackoverflow.com-Posts.7z
python3 ./src/preprocessor/preprocess.py Posts.xml train 0 50000 true false
python3 ./src/preprocessor/preprocess.py Posts.xml dev 50000 25000 true false
python3 ./src/preprocessor/preprocess.py Posts.xml eval 75000 25000 true false
python3 ./src/preprocessor/preprocess.py Posts.xml train 0 50000 true true
python3 ./src/preprocessor/preprocess.py Posts.xml dev 50000 25000 true true
python3 ./src/preprocessor/preprocess.py Posts.xml eval 75000 25000 true true
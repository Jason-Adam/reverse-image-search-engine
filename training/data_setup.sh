#!/usr/bin/env sh


set -e;

echo "downloading data if not present"
python training/download.py

echo "extracting tarball"
tar -xvf 101_ObjectCategories.tar.gz && \
    mv 101_ObjectCategories caltech101 && \
    rm -rf caltech101/BACKGROUND_Google && \
    rm 101_ObjectCategories.tar.gz;

echo "finishied!"

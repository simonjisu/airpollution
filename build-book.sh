#!/bin/bash
git add .
git commit -m 'update'
git push origin main
git checkout gh-pages
git pull origin main
jupyter-book build book --path-output ./
mv _build docs
git add .
git commit -m 'update book'
git push origin gh-pages
git checkout main

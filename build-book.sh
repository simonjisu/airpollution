#!/bin/bash
git checkout gh-pages
git rebase origin main
git merge origin main
jupyter-book build book --path-output ./docs
git add .
git commit -m 'update book'
git push origin gh-pages
git checkout main

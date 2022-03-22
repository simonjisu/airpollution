#!/bin/bash
jupyter-book build book --path-output ./
ghp-import -n -p -f _build/html
git add .
git commit -m 'jupyter-book publish'
git push origin main
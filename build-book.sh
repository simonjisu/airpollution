#!/bin/bash
jupyter-book build book --path-output ./
ghp-import -n -p -f _build/html
import gdown
from pathlib import Path

# database
link_id = '1UylI7ZvnL4OTvumeBMnIMR7RGd9Rak2S'
url = f'https://drive.google.com/uc?id={link_id}'
data_path = Path() / 'data'
if not data_path.exists():
    data_path.mkdir()
with (data_path / 'airpollution.db').open('wb') as file:
    gdown.download(url, file, quiet=False)
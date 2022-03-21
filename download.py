import gdown
from pathlib import Path
url = 'https://drive.google.com/uc?id=1ppI6B488QgYSC6pXI2z7TVDdMYPGRhyx'
data_path = Path() / 'data'
if not data_path.exists():
    data_path.mkdir()
output = data_path / 'data.zip'
gdown.download(url, output, quiet=False)
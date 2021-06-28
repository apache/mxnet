from pathlib import Path
import glob, json

constructor = {"cell_type": "markdown", "metadata": {}}
prefix = "Run this notebook online:[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deepjavalibrary/djl/master?filepath="

for file in Path('.').glob('**/*.ipynb'):
    with open(file, mode= "r", encoding= "utf-8") as f:
        data = json.loads(f.read())
    with open(file, 'w') as writer:
        constructor["source"] = [prefix + str(file) + ")"]
        data["cells"].insert(0, constructor)
        writer.write(json.dumps(data))

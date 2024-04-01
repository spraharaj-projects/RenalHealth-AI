import os
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'renalhealth_ai'
package_name = 'cnn_classifier'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/{package_name}/__init__.py",
    f"{project_name}/{package_name}/components/__init__.py",
    f"{project_name}/{package_name}/utils/__init__.py",
    f"{project_name}/{package_name}/config/__init__.py",
    f"{project_name}/{package_name}/config/configuration.py",
    f"{project_name}/{package_name}/pipeline/__init__.py",
    f"{project_name}/{package_name}/entity/__init__.py",
    f"{project_name}/{package_name}/constant/__init__.py",
    "config/config.yml",
    "dvc.yml",
    "params.yml",
    "research/trials.ipynb",
    "requirements.txt",
    "setup.py",
]

for file_path in list_of_files:
    file_dir, file_name = os.path.split(Path(file_path))

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)) == 0:
        with open(file_path, "w") as f:
            logging.info(f"Creating empty file: {file_path}")

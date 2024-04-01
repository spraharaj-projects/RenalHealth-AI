import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "RenalHealth-AI"
AUTHOR_USER_NAME = "spraharaj-projects"
SRC_REPO = "cnn_classifier"
AUTHOR_EMAIL = "supreet.praharaj.pro@gmail.com"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="CNN Classifier for Kidney Diseases",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "renalhealth_ai"},
    packages=setuptools.find_packages(where="renalhealth_ai")
)

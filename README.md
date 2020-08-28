# Fastaudio
> An audio module for fastai v2. We want to help you build audio machine learning applications while minimizing the need for audio domain expertise. Currently under development.

## Install

In the future we will offer conda and pip installs, but as the code is rapidly changing, we recommend that only those interested in contributing and experimenting install for now. Everyone else should use [Fastai audio v1](https://github.com/mogwai/fastai_audio)

To install:

```
git clone https://github.com/fastaudio/fastaudio.git
conda env create -f fastaudio/environment.yaml
cd fastaudio && pip install .
```

If you plan on contributing to the library instead, you will need to do a editable install:

```
git clone https://github.com/fastaudio/fastaudio.git
conda env create -f fastaudio/environment.yaml
cd fastaudio
pip install -e .[dev,testing]
pre-commit install
```

# Contributing to the library

We are looking for contributors of all skill levels. If you don't have time to contribute, please at least reach out and give us some feedback on the library.

### How to contribute
Create issues, write documentation, suggest/add features, submit PRs. We are open to anything.


## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

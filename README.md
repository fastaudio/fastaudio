![Tests](https://github.com/fastaudio/fastaudio/workflows/Python%20package/badge.svg)
[![Docs](https://img.shields.io/badge/docs-latest-green)](https://fastaudio.github.io/)
[![codecov](https://codecov.io/gh/fastaudio/fastaudio/branch/master/graph/badge.svg)](https://codecov.io/gh/fastaudio/fastaudio)


# Fastaudio
> An audio module for fastai v2. We want to help you build audio machine learning applications while minimizing the need for audio domain expertise. Currently under development.

## Install

In the future we will offer conda and pip installs, but as the code is rapidly changing, we recommend that only those interested in contributing and experimenting install for now. Everyone else should use [Fastai audio v1](https://github.com/mogwai/fastai_audio)

---

Install using pip:

```
pip install git+https://github.com/fastaudio/fastaudio.git
```

---

If you plan on **contributing** to the library instead, you will need to do a editable install:

```
# Optional step if using conda
conda create -n fastaudio python=3.7
conda activate fastaudio
```

```
# Editable install
git clone https://github.com/fastaudio/fastaudio.git
cd fastaudio
pip install -e .[dev,testing]
pre-commit install
```

# Testing
To run the tests and verify everythig is working, you'll need to create an environment like above:

```
git clone git@github.com:fastaudio/fastaudio
pip install .[dev,testing]
pytest
```

# Contributing to the library

We are looking for contributors of all skill levels. If you don't have time to contribute, please at least reach out and give us some feedback on the library.

Make sure that you have activated the environment that you used `pre-commit install` in so that pre-commit knows where to run the git hooks.

### How to contribute
Create issues, write documentation, suggest/add features, submit PRs. We are open to anything.

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

## Citation

If you used this library in any research, please cite us.

```
@misc{coultas_blum_scart_bracco_2020,
 title={Fastaudio},
 url={https://github.com/fastaudio/fastaudio},
 journal={GitHub},
 author={Coultas Blum, Harry A and Scart, Lucas G. and Bracco, Robert},
 year={2020},
 month={Aug}
}
```

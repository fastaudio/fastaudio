import papermill as pm
from glob import glob


def test_can_run_notebooks():
    # Search for all notebooks in directory
    notebooks = glob("**/*.ipynb")
    for nb in notebooks:
        try:
            pm.execute_notebook(nb, "/dev/null")
        except Exception as e:
            # Which notebook caused the error
            raise Exception(nb, e)

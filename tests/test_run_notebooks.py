from glob import glob

import nbformat
from nbconvert import PythonExporter


def convert_notebook(notebookPath):
    """
    This will convert a notebook into a script
    that we can run
    """
    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    lines = source.split("\n")
    nlines = []

    # Remove Magic %% lines
    # Remove !ls commands
    for i in range(len(lines)):
        if "get_ipython" in lines[i]:
            continue
        stripped = lines[i].strip()
        if len(stripped) > 0 and stripped[0] == "!":
            continue
        nlines.append(lines[i])
    source = "\n".join(nlines)
    print(source)
    return source, meta


def test_can_run_notebooks():
    # Search for all notebooks in directory
    notebooks = glob("**/*.ipynb")
    for nb in notebooks:
        src, meta = convert_notebook(nb)
        try:
            exec(src)
        except Exception as e:
            # Which notebook caused the error
            print("Notebook: ", nb)
            raise e

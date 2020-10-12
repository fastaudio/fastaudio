from glob import glob

import nbformat
from nbconvert import PythonExporter


def convertNotebook(notebookPath):
    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    lines = source.split("\n")
    nlines = []
    for i in range(len(lines)):
        if "get_ipython" in lines[i]:
            continue
        stripped = lines[i].strip()
        if len(stripped) > 0 and stripped[0] == "!":
            continue
        nlines.append(lines[i])
    source = "\n".join(nlines)
    return source, meta


def test_can_run_notebooks():
    notebooks = glob("../**/*.ipynb")
    for nb in notebooks:
        src, meta = convertNotebook(nb)
        try:
            exec(src)
        except Exception as e:
            print("Notebook: ", nb)
            raise e

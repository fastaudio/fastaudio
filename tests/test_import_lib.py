import fastaudio


def test_import_lib():
    """This is a dummy test, just to check
    if the lib can be successfully imported.
    """
    assert fastaudio.__version__ is not None

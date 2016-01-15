import os


def ensure_dir(f):
    """
      Ensure that a directory exists. Create if it doesn't
    """
    d = os.path.dirname(f)
    if d == "":
        d = f
    if not os.path.exists(d):
        os.makedirs(d)
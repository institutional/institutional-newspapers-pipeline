import os


def make_dirs() -> None:
    """
    Creates target dirs as needed.
    Throws if any of the target destinations cannot be written.
    """
    import const

    for name in dir(const):
        if name.endswith("_DIR_PATH"):
            os.makedirs(getattr(const, name), exist_ok=True)

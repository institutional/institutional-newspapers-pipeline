import peewee

import utils


def create_tables() -> bool:
    """Lists models and automatically creates database tables if needed."""
    import models

    available_models = [
        getattr(models, name)
        for name in dir(models)
        if isinstance(getattr(models, name, None), type)
        and issubclass(getattr(models, name), peewee.Model)
        and getattr(models, name) is not peewee.Model
    ]

    with utils.get_db() as db:
        db.create_tables(available_models)

    return True

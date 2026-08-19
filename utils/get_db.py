import os

import peewee

database_proxy = peewee.DatabaseProxy()

_init_pid = None
""" Keeps track of the pid from which the connection was initialized. """


def get_db() -> peewee.DatabaseProxy:
    """
    Process-safe access to the database.

    Returns an active database proxy.

    Automatically recreates connection:
    - Not available
    - `get_db()` is called from a subprocess/fork.
    """
    global _init_pid
    current_pid = os.getpid()

    if database_proxy.obj is None or _init_pid != current_pid or database_proxy.obj.is_closed():
        from const import DATABASE_FILEPATH

        db = peewee.SqliteDatabase(
            str(DATABASE_FILEPATH),
            pragmas={
                "journal_mode": "wal",
                "wal_autocheckpoint": 10000,
                "synchronous": "normal",
                "cache_size": -1000000,
                "foreign_keys": 1,
                "busy_timeout": 60000,
                "mmap_size": 68719476736,
                "temp_store": "memory",
                "page_size": 8192,
            },
        )

        try:
            db.connect()
            database_proxy.initialize(db)
            _init_pid = current_pid
        except Exception as err:
            raise ConnectionError(f"Could not connect to SQLite at {DATABASE_FILEPATH}.") from err

    return database_proxy

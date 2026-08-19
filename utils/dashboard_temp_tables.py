import peewee

import const


def create_crop_temp_table(
    db: peewee.Database,
    batch_ids: list[int],
) -> None:
    """Populates a temporary table with crop IDs belonging to the given batches."""
    db.execute_sql(
        "CREATE TEMP TABLE IF NOT EXISTS _dash_crops "
        "(crop_id INTEGER PRIMARY KEY)"
    )
    db.execute_sql("DELETE FROM _dash_crops")

    for i in range(0, len(batch_ids), const.DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = batch_ids[i : i + const.DB_IN_CLAUSE_CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        db.execute_sql(
            "INSERT OR IGNORE INTO _dash_crops (crop_id) "
            "SELECT c.id FROM crop c "
            "INNER JOIN pipeline_batch_item pbi ON c.pipeline_batch_item_id = pbi.id "
            f"WHERE pbi.pipeline_batch_id IN ({placeholders})",
            chunk,
        )


def create_crop_year_temp_table(
    db: peewee.Database,
    batch_ids: list[int],
) -> None:
    """Populates a temporary table with crop IDs and their associated issue year."""
    db.execute_sql(
        "CREATE TEMP TABLE IF NOT EXISTS _dash_crops_year "
        "(crop_id INTEGER PRIMARY KEY, year INTEGER)"
    )
    db.execute_sql("DELETE FROM _dash_crops_year")

    for i in range(0, len(batch_ids), const.DB_IN_CLAUSE_CHUNK_SIZE):
        chunk = batch_ids[i : i + const.DB_IN_CLAUSE_CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        db.execute_sql(
            "INSERT OR IGNORE INTO _dash_crops_year (crop_id, year) "
            "SELECT c.id, i.year FROM crop c "
            "INNER JOIN pipeline_batch_item pbi ON c.pipeline_batch_item_id = pbi.id "
            "INNER JOIN scan s ON c.scan_id = s.id "
            "INNER JOIN issue i ON s.issue_id = i.id "
            f"WHERE pbi.pipeline_batch_id IN ({placeholders})",
            chunk,
        )


def create_crop_pre1931_temp_table(db: peewee.Database) -> int:
    """Populates a temporary table with crop IDs from issues published before 1931.

    Returns the number of rows inserted (0 means no pre-1931 data exists).
    """
    db.execute_sql(
        "CREATE TEMP TABLE IF NOT EXISTS _dash_crops_pre1931 "
        "(crop_id INTEGER PRIMARY KEY)"
    )
    db.execute_sql("DELETE FROM _dash_crops_pre1931")
    db.execute_sql(
        "INSERT INTO _dash_crops_pre1931 (crop_id) "
        "SELECT crop_id FROM _dash_crops_year WHERE year < 1931"
    )
    row = db.execute_sql("SELECT COUNT(*) FROM _dash_crops_pre1931").fetchone()
    return row[0] if row else 0


def drop_temp_tables(db: peewee.Database) -> None:
    """Drops the temporary dashboard tables."""
    db.execute_sql("DROP TABLE IF EXISTS _dash_crops")
    db.execute_sql("DROP TABLE IF EXISTS _dash_crops_year")
    db.execute_sql("DROP TABLE IF EXISTS _dash_crops_pre1931")

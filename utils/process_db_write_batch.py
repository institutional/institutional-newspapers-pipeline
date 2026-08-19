import peewee


def process_db_write_batch(
    model: peewee.Model,
    entries_to_create: list[peewee.Model] | None = None,
    entries_to_update: list[peewee.Model] | None = None,
    fields_to_update: list[peewee.Field] | None = None,
) -> bool:
    """
    Processes a batch of database create/update operations.
    Empties `entries_to_create` and `entries_to_update` in place.
    """
    if entries_to_create is None:
        entries_to_create = []

    if entries_to_update is None:
        entries_to_update = []

    if fields_to_update is None:
        fields_to_update = []

    num_fields = len(model._meta.sorted_fields)
    batch_size = max(1, 999 // num_fields)

    if entries_to_create:
        model.bulk_create(
            entries_to_create,
            batch_size=batch_size,
        )
        entries_to_create.clear()

    if entries_to_update:
        model.bulk_update(
            entries_to_update,
            fields=fields_to_update,
            batch_size=batch_size,
        )
        entries_to_update.clear()

    return True

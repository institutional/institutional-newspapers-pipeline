import peewee

import utils
from models import Issue, PipelineBatch


class PipelineBatchItem(peewee.Model):
    """
    `pipeline_batch_item` table: Keeps track of individual items within a pipeline run batch.
    """

    class Meta:
        table_name = "pipeline_batch_item"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    pipeline_batch = peewee.ForeignKeyField(
        model=PipelineBatch,
        field="id",
        index=True,
    )

    issue = peewee.ForeignKeyField(model=Issue, field="id", index=True)

import peewee

import utils
from models import PipelineRun, Issue


class PipelineBatch(peewee.Model):
    """
    `pipeline_batch` table: Keeps track of pipeline run batches.
    """

    class Meta:
        table_name = "pipeline_batch"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    pipeline_run = peewee.ForeignKeyField(
        model=PipelineRun,
        field="id",
        index=True,
    )

    node_name = peewee.CharField(max_length=128, null=True, index=False)  # Assigned at run time

    created_date = peewee.DateTimeField(null=True)

    started_date = peewee.DateTimeField(null=True)

    ended_date = peewee.DateTimeField(null=True)

    has_crashed = peewee.BooleanField(default=False)

    @property
    def items(self) -> list:
        """
        Returns a sorted list of PipelineBatchItems instances from this batch.
        """
        from models import PipelineBatchItem, Issue

        items = []

        for pipeline_batch_item in (
            PipelineBatchItem.select(PipelineBatchItem, Issue)
            .where(PipelineBatchItem.pipeline_batch == self.id)
            .join(Issue)
            .order_by(PipelineBatchItem.id)
            .iterator()
        ):
            items.append(pipeline_batch_item)

        return items

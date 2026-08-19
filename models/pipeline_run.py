import peewee

import utils
from const import CORPORA


class PipelineRun(peewee.Model):
    """
    `pipeline_run` table: Keeps track of pipeline runs.
    """

    class Meta:
        table_name = "pipeline_run"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    corpus = peewee.CharField(
        index=True,
        choices=[(c, c) for c in CORPORA],
    )

    items_total = peewee.IntegerField(null=False)

    items_per_batch = peewee.IntegerField(null=False)

    batches_total = peewee.IntegerField(null=False)

    created_date = peewee.DateTimeField(null=True)

    @property
    def batches(self) -> list:
        """
        Returns a sorted list of PipelineBatch instances from this run.
        """
        from models import PipelineBatch

        batches = []

        for pipeline_batch in (
            PipelineBatch.select()
            .where(PipelineBatch.pipeline_run == self.id)
            .order_by(PipelineBatch.id)
            .iterator()
        ):
            batches.append(pipeline_batch)

        return batches

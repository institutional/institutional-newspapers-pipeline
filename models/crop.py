import peewee

import utils
from models import Scan, JSONArrayField


class Crop(peewee.Model):
    """
    `crop` table: Keeps track of individual newspaper scan crops
    """

    class Meta:
        table_name = "crop"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    scan = peewee.ForeignKeyField(model=Scan, field=Scan.id)

    pipeline_batch_item = peewee.DeferredForeignKey("PipelineBatchItem", null=True)

    bbox_xyxy = JSONArrayField(null=True)

    confidence_score = peewee.FloatField(null=True)

    reading_order = peewee.IntegerField(null=True)

    width = peewee.IntegerField(null=True)

    height = peewee.IntegerField(null=True)

    @property
    def cache_key(self) -> str:
        """Returns a cache key for the current crop."""
        key = f"{self.scan.issue.corpus}__"
        key += f"{self.scan.issue.archive_filename}__"
        key += f"{self.scan.scan_filename}__"
        key += f"{self.id}"
        return key

import peewee
import PIL

import utils
from const import CORPORA, CPUS_LIMIT


class Issue(peewee.Model):
    """
    `issue` table: Keeps track of newspaper issues and their metadata
    """

    class Meta:
        table_name = "issue"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    corpus = peewee.CharField(
        index=True,
        choices=[(c, c) for c in CORPORA],
    )

    archive_filename = peewee.CharField(
        max_length=512,
        null=False,
        index=True,
    )

    archive_size_bytes = peewee.BigIntegerField(
        null=True,
    )

    newspaper_id = peewee.CharField(
        max_length=256,
        null=True,
    )

    newspaper_id_type = peewee.CharField(
        max_length=256,
        null=True,
    )

    edition_slug = peewee.CharField(
        max_length=256,
        null=True,
    )

    edition_slug_type = peewee.CharField(
        max_length=256,
        null=True,
    )

    title = peewee.CharField(
        max_length=256,
        null=True,
        index=False,
    )

    city = peewee.CharField(max_length=256, null=True)

    state = peewee.CharField(max_length=32, null=True)

    country = peewee.CharField(max_length=128, null=True)

    publisher = peewee.CharField(max_length=512, null=True)

    year = peewee.IntegerField(null=True, index=True)

    month = peewee.IntegerField(null=True)

    day = peewee.IntegerField(null=True)

    edition_number = peewee.IntegerField(null=True)

    year_start = peewee.IntegerField(null=True)

    year_end = peewee.IntegerField(null=True)

    language = peewee.CharField(max_length=16, null=True)

    loc_access_restricted = peewee.BooleanField(null=True)

    @property
    def cache_key(self) -> str:
        """Returns a cache key for the current issue."""
        return f"{self.corpus}__{self.archive_filename}"

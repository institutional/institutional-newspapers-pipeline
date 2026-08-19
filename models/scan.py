from pathlib import Path
import os
import gzip
import tarfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import peewee
import PIL
import numpy as np
import cv2

import utils
from models import Issue
import const


class Scan(peewee.Model):
    """
    `scan` table: Keeps track of individual newspaper scans
    """

    class Meta:
        table_name = "scan"
        database = utils.get_db()

    id = peewee.PrimaryKeyField()

    issue = peewee.ForeignKeyField(model=Issue, field=Issue.id)

    scan_filename = peewee.CharField(
        max_length=512,
        null=False,
        index=True,
    )

    width = peewee.IntegerField(null=True)

    height = peewee.IntegerField(null=True)

    phash = peewee.CharField(
        max_length=512,
        null=False,
    )

    @property
    def cache_key(self) -> str:
        """Returns a cache key for the current scan."""
        return f"{self.issue.corpus}__{self.issue.archive_filename}__{self.scan_filename}"

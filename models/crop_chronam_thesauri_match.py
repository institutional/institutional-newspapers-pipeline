import peewee
from playhouse.sqlite_ext import JSONField

import utils
from models import Crop


class CropChronamThesauriMatch(peewee.Model):
    """
    `crop_chronam_thesauri_match` table: Tracks matches of Chronicling America thesauri
    terms in individual newspaper crop OCR text (both Tesseract and VLM sources).

    The `*_matches` JSON fields store nested dicts grouped by category:
    `{"category": {"term": count, ...}, ...}`.
    `None` means no text was available; empty `{}` means text was present but no terms matched.
    """

    class Meta:
        table_name = "crop_chronam_thesauri_match"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    tesseract_matches = JSONField(default=None, null=True)
    tesseract_match_count = peewee.IntegerField(null=True)
    tesseract_term_count = peewee.IntegerField(null=True)

    vlm_matches = JSONField(default=None, null=True)
    vlm_match_count = peewee.IntegerField(null=True)
    vlm_term_count = peewee.IntegerField(null=True)

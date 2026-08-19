import peewee

import utils
from models import Crop


class CropLanguage(peewee.Model):
    """
    `crop_language` table: Detected language of individual newspaper crops' OCR text.
    """

    class Meta:
        table_name = "crop_language"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    language_code = peewee.CharField(max_length=3, null=True)

    confidence_score = peewee.FloatField(null=True)

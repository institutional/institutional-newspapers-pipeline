import peewee
from playhouse.sqlite_ext import JSONField

import utils
from models import Crop


class CropOCR(peewee.Model):
    """
    `crop_ocr` table: Keeps track of the OCR data of individual newspaper crops
    """

    class Meta:
        table_name = "crop_ocr"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    tesseract_text = peewee.TextField(null=True)

    tesseract_metadata = JSONField(default=None, null=True)

    vlm_text = peewee.TextField(null=True)

    vlm_metadata = JSONField(default=None, null=True)

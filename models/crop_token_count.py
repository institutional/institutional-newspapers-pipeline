import peewee

import utils
from models import Crop


class CropTokenCount(peewee.Model):
    """
    `crop_token_count` table: Token counts for OCR text of individual newspaper crops.
    """

    class Meta:
        table_name = "crop_token_count"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    tesseract_token_count = peewee.IntegerField(null=True)

    vlm_token_count = peewee.IntegerField(null=True)

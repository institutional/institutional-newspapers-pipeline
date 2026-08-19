import peewee

import utils
from models import Crop


class CropClassification(peewee.Model):
    """
    `crop_classification` table: Keeps track of the classification data of individual newspaper crops
    """

    class Meta:
        table_name = "crop_classification"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    image_category = peewee.CharField(max_length=64, null=True)

    image_confidence_score = peewee.FloatField(null=True)

    text_category = peewee.CharField(max_length=64, null=True)

    text_confidence_score = peewee.FloatField(null=True)

    final_category = peewee.CharField(max_length=64, null=True)

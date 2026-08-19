import peewee

import utils
from models import Crop, JSONArrayField


class CropNER(peewee.Model):
    """
    `crop_ner` table: Keeps track of the classification data of individual newspaper crops
    """

    class Meta:
        table_name = "crop_ner"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    per_entities = JSONArrayField()

    per_confidence_scores = JSONArrayField()

    loc_entities = JSONArrayField()

    loc_confidence_scores = JSONArrayField()

    org_entities = JSONArrayField()

    org_confidence_scores = JSONArrayField()

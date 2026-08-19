import peewee

import utils
from models import Crop, JSONArrayField


class CropSubject(peewee.Model):
    """
    `crop_subject` table: Keeps track of the classification data of individual newspaper crops
    """

    class Meta:
        table_name = "crop_subject"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    ranked_labels = JSONArrayField()

    scores = JSONArrayField()

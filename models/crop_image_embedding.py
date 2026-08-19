import peewee

import utils
from models import Crop, JSONArrayField


class CropImageEmbedding(peewee.Model):
    """
    `crop_image_embedding` table: DINOv2-small image embeddings for individual newspaper crops,
    used for visual similarity search.
    """

    class Meta:
        table_name = "crop_image_embedding"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    embedding = JSONArrayField(null=True)

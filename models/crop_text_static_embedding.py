import peewee

import utils
from models import Crop, JSONArrayField


class CropTextStaticEmbedding(peewee.Model):
    """
    `crop_text_static_embedding` table: Static text embeddings for individual newspaper crops,
    generated using model2vec from flattened VLM OCR text.
    """

    class Meta:
        table_name = "crop_text_static_embedding"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    vlm_embedding = JSONArrayField(null=True)

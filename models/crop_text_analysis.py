import peewee

import utils
from models import Crop


class CropTextAnalysis(peewee.Model):
    """
    `crop_text_analysis` table: Text analysis metrics for OCR text of individual newspaper crops.
    Stores metrics for both tesseract and VLM OCR sources.
    """

    class Meta:
        table_name = "crop_text_analysis"
        database = utils.get_db()

    crop = peewee.ForeignKeyField(model=Crop, field=Crop.id, primary_key=True, on_delete="CASCADE")

    tesseract_tokenizability_score = peewee.FloatField(null=True)
    tesseract_char_count = peewee.IntegerField(null=True)
    tesseract_word_count = peewee.IntegerField(null=True)
    tesseract_word_count_unique = peewee.IntegerField(null=True)
    tesseract_word_type_token_ratio = peewee.FloatField(null=True)
    tesseract_sentence_count = peewee.IntegerField(null=True)
    tesseract_sentence_count_unique = peewee.IntegerField(null=True)

    vlm_tokenizability_score = peewee.FloatField(null=True)
    vlm_char_count = peewee.IntegerField(null=True)
    vlm_word_count = peewee.IntegerField(null=True)
    vlm_word_count_unique = peewee.IntegerField(null=True)
    vlm_word_type_token_ratio = peewee.FloatField(null=True)
    vlm_sentence_count = peewee.IntegerField(null=True)
    vlm_sentence_count_unique = peewee.IntegerField(null=True)
    vlm_has_table = peewee.BooleanField(null=True)
    vlm_has_markdown = peewee.BooleanField(null=True)

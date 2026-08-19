import json

import peewee


class JSONArrayField(peewee.TextField):
    """Stores Python lists as JSON text. Drop-in replacement for PostgreSQL ArrayField."""

    def db_value(self, value: list | None) -> str | None:
        if value is not None:
            return json.dumps(value)
        return None

    def python_value(self, value: str | None) -> list | None:
        if value is not None:
            return json.loads(value)
        return None


from .issue import Issue
from .scan import Scan
from .crop import Crop
from .crop_ocr import CropOCR
from .crop_token_count import CropTokenCount
from .crop_language import CropLanguage
from .crop_text_analysis import CropTextAnalysis
from .crop_chronam_thesauri_match import CropChronamThesauriMatch
from .crop_text_static_embedding import CropTextStaticEmbedding
from .crop_image_embedding import CropImageEmbedding
from .crop_classification import CropClassification
from .crop_subject import CropSubject
from .crop_ner import CropNER
from .pipeline_run import PipelineRun
from .pipeline_batch import PipelineBatch
from .pipeline_batch_item import PipelineBatchItem

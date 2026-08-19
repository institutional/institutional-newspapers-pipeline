from .clear_cache import clear_cache
from .build import build
from .warmup_ocr_vlm import warmup_ocr_vlm
from .cache_models import cache_models

import click


@click.group("system")
def system():
    pass


system.add_command(clear_cache)
system.add_command(build)
system.add_command(warmup_ocr_vlm)
system.add_command(cache_models)

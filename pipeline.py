from dotenv import load_dotenv

load_dotenv()

import sys

import click
from loguru import logger

import utils
from commands.system import system
from commands.orchestration import orchestration
from commands.steps import steps
from commands.analysis import analysis
from commands.peek import peek
from commands.export import export

utils.check_env()
utils.make_dirs()


@click.group()
@click.option(
    "--verbose",
    is_flag=True,
    help="If set, includes DEBUG-level statements in log output.",
)
def cli(verbose: bool):
    from ultralytics import settings
    from PIL import Image

    # Disable Pillow's DecompressionBombWarning
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Always disable Ultralytics telemetry
    settings.update({"sync": False})

    # Enable logger
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)

    # Always try to create tables
    utils.create_tables()


cli.add_command(system)
cli.add_command(orchestration)
cli.add_command(steps)
cli.add_command(analysis)
cli.add_command(peek)
cli.add_command(export)

if __name__ == "__main__":
    cli()

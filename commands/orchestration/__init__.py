import click

from .prepare import prepare
from .status import status
from .execute import execute


@click.group("orchestration")
def orchestration():
    pass


orchestration.add_command(prepare)
orchestration.add_command(status)
orchestration.add_command(execute)

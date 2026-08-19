import click

from .logs import logs
from .dashboard import dashboard


@click.group("analysis")
def analysis():
    pass


analysis.add_command(logs)
analysis.add_command(dashboard)

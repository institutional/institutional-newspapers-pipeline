import click
from humanize import intcomma, naturaltime

from models import PipelineRun


@click.command("status")
def status():
    """Lists all pipeline runs and their batches with current status, node assignment, and timing."""
    for run in PipelineRun.select().order_by(PipelineRun.id).iterator():
        click.echo(80 * "-")
        click.echo(f"RUN #{run.id}")
        click.echo(80 * "-")

        click.echo(f"- Created {naturaltime(run.created_date)}")
        click.echo(f"- Items: {intcomma(run.items_total)}")
        click.echo(f"- Total batches: {intcomma(run.batches_total)}")
        click.echo(f"- Items per batch: {intcomma(run.items_per_batch)}")

        for batch in run.batches:
            message = ""
            status = "PENDING"

            if batch.started_date and not batch.ended_date:
                status = "RUNNING/LOCKED"

            if batch.has_crashed:
                status = "CRASHED"

            if batch.started_date and batch.ended_date and not batch.has_crashed:
                status = "COMPLETED"

            click.echo(40 * "-")
            message = f"-- BATCH #{batch.id} ({status})\n"
            message += f"--- Node: {batch.node_name}\n"

            if batch.started_date:
                message += f"--- Started {naturaltime(batch.started_date)}\n"

            if batch.ended_date:
                message += f"--- Ended {naturaltime(batch.ended_date)}\n"

            click.echo(message.strip())

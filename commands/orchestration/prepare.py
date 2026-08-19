import random
import traceback
import math
from datetime import datetime, timezone

import click
from loguru import logger
from humanize import intcomma

from utils import process_db_write_batch
from models import Issue, PipelineRun, PipelineBatch, PipelineBatchItem
from const import CORPORA


@click.command("prepare")
@click.option(
    "--corpus",
    type=click.Choice(CORPORA),
    required=True,
    help="Corpus to create the run for.",
)
@click.option(
    "--offset",
    type=int,
    required=False,
    default=None,
    help="Allows for creating a run on a subset of issues. Issues are ordered by id asc.",
)
@click.option(
    "--limit",
    type=int,
    required=False,
    default=None,
    help="Allows for creating a run on a subset of issues. Issues are ordered by id asc.",
)
@click.option(
    "--items-per-batch",
    type=click.IntRange(10, 1000),
    default=100,
)
@click.option(
    "--append-mode",
    is_flag=True,
    help="If set, will create a run for issues that are not part of any other run.",
)
@click.option(
    "--shuffle",
    is_flag=True,
    help="If set, randomly shuffles issues before applying --offset/--limit and splitting into batches.",
)
def prepare(
    corpus: str,
    offset: int,
    limit: int,
    items_per_batch: int,
    append_mode: bool,
    shuffle: bool,
):
    """
    Creates a pipeline run and its batches.
    Splits issues into fixed-size batches and returns a run identifier that can be passed to `orchestration execute`.
    """
    items_total = Issue.select().where(Issue.corpus == corpus).offset(offset).limit(limit).count()
    batches_total = 0
    issues_batches = []
    issues_exclusion_list = set()

    #
    # If `--append-mode` exclude items that were already processed
    #
    if append_mode:
        logger.info("Listing items to exclude from run (--append-mode)")

        issues_exclusion_list = {
            row.id for row in PipelineBatchItem.select(PipelineBatchItem.issue).distinct()
        }

        items_total = items_total - len(issues_exclusion_list)

    if items_total < 1:
        logger.info("No items left to process.")
        click.get_current_context().exit(0)

    batches_total = int(math.ceil(items_total / items_per_batch))

    # Ask for confirmation
    logger.info(f"Total issues: {intcomma(items_total)}")
    logger.info(f"Issues per batch: {intcomma(items_per_batch)}")
    logger.info(f"Total batches: {intcomma(batches_total)}")

    if not click.confirm("Proceed?"):
        click.get_current_context().exit(0)

    #
    # Create new pipeline run
    #
    pipeline_run = PipelineRun(
        corpus=corpus,
        items_total=items_total,
        items_per_batch=items_per_batch,
        batches_total=batches_total,
        created_date=datetime.now(timezone.utc),
    )

    pipeline_run.save()

    logger.info(f"Pipeline run id: {pipeline_run.id}")

    #
    # Create and save pipeline batches
    #
    logger.info(f"Creating batches for run {pipeline_run.id} ...")

    # Split issues list into batches of volumes
    issues_batches.append([])

    if shuffle:
        issue_ids = [
            row.id
            for row in Issue.select(Issue.id).where(Issue.corpus == corpus).order_by(Issue.id)
        ]
        random.shuffle(issue_ids)
        issue_ids = issue_ids[offset:None if limit is None else (offset or 0) + limit]
    else:
        issue_ids = [
            row.id
            for row in Issue.select(Issue.id)
            .where(Issue.corpus == corpus)
            .offset(offset)
            .limit(limit)
            .order_by(Issue.id)
        ]

    for issue_id in issue_ids:
        issues_batch = issues_batches[-1]

        # If --append-mode: skip issues that are already part of a run
        if append_mode and issue_id in issues_exclusion_list:
            continue

        if len(issues_batch) >= items_per_batch:
            issues_batches.append([])
            issues_batch = issues_batches[-1]

        issues_batch.append(issue_id)

    # Create pipeline batches and pipeline batches items for each volumes batch
    entries_to_create = []

    for issues_batch in issues_batches:

        pipeline_batch = PipelineBatch(
            pipeline_run=pipeline_run,
            created_date=datetime.now(timezone.utc),
        )

        pipeline_batch.save()

        for issue_id in issues_batch:
            pipeline_batch_item = PipelineBatchItem(
                pipeline_batch=pipeline_batch,
                issue=issue_id,
            )

            entries_to_create.append(pipeline_batch_item)

    process_db_write_batch(PipelineBatchItem, entries_to_create)

    logger.info(f"Pipeline run ready. Pipeline run id: {pipeline_run.id}")

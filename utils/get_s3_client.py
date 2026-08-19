import os
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

if TYPE_CHECKING:
    from botocore.client import BaseClient


def get_s3_client(target: str, *, max_pool_connections: int | None = None) -> "BaseClient":
    """
    Returns an S3 client connected to either:
    - one of the corpus upstream buckets
    - the downstream "OUTPUT" bucket
    - the "RELEASE" bucket for published datasets

    `max_pool_connections` overrides the botocore connection-pool size; callers driving many
    concurrent requests through one client should raise it above the botocore default of 10.
    """
    from const import CORPORA

    if target not in CORPORA and target not in ("OUTPUT", "RELEASE"):
        raise Exception(f"{target} is not a valid target")

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(f"{target}_S3_ENDPOINT"),
        aws_access_key_id=os.environ.get(f"{target}_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get(f"{target}_S3_SECRET_ACCESS_KEY"),
        config=Config(
            region_name=os.environ.get(f"{target}_S3_REGION", "auto"),
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            max_pool_connections=max_pool_connections or 10,
        ),
    )

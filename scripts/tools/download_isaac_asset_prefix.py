#!/usr/bin/env python3
"""Mirror a public Isaac asset prefix from NVIDIA's S3 bucket."""

from __future__ import annotations

import argparse
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BUCKET_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
XML_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="S3 key prefix to mirror.")
    parser.add_argument("--output", type=Path, required=True, help="Destination root.")
    return parser.parse_args()


def list_objects(prefix: str):
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = f"{BUCKET_URL}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url) as response:
            root = ET.parse(response).getroot()
        for entry in root.findall("s3:Contents", XML_NAMESPACE):
            key = entry.findtext("s3:Key", namespaces=XML_NAMESPACE)
            size = int(entry.findtext("s3:Size", namespaces=XML_NAMESPACE) or 0)
            if key:
                yield key, size
        if root.findtext("s3:IsTruncated", namespaces=XML_NAMESPACE) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=XML_NAMESPACE)
        if not token:
            raise RuntimeError("S3 listing was truncated without a continuation token")


def main() -> None:
    args = parse_args()
    count = 0
    for key, size in list_objects(args.prefix):
        destination = args.output / Path(key)
        if destination.is_file() and destination.stat().st_size == size:
            count += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        url = f"{BUCKET_URL}/{urllib.parse.quote(key, safe='/')}"
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size != size:
            raise RuntimeError(f"size mismatch for {key}: {temporary.stat().st_size} != {size}")
        os.replace(temporary, destination)
        count += 1
        print(f"[{count}] {key}", flush=True)
    print(f"Mirrored {count} objects under {args.output}")


if __name__ == "__main__":
    main()

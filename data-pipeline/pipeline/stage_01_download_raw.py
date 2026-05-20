"""Скачивает всё из бакета в data/raw - csv с объявлениями и картинки"""

import os
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import ENV_FILE, RAW_DIR


def main():
    load_dotenv(ENV_FILE)

    bucket = os.environ['RUSTFS_BUCKET']
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['RUSTFS_ENDPOINT'].rstrip('/'),
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for obj in s3.list_objects_v2(Bucket=bucket).get('Contents', []):
        key = obj['Key']
        dest = RAW_DIR / Path(key).name
        print(f'качаю {key} -> {dest}')
        s3.download_file(bucket, key, str(dest))


if __name__ == '__main__':
    main()

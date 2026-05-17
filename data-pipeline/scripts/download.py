'''Скачивает все файлы из бакета RustFS в data/raw'''

import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = PROJECT_ROOT / 'data' / 'raw'


def main():
    load_dotenv(PROJECT_ROOT / '.env')

    bucket = os.environ['RUSTFS_BUCKET']
    s3 = boto3.client(
        's3',
        endpoint_url=os.environ['RUSTFS_ENDPOINT'].rstrip('/'),
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for obj in s3.list_objects_v2(Bucket=bucket).get('Contents', []):
        key = obj['Key']
        dest = DEST_DIR / Path(key).name
        print(f'Качаю {key} -> {dest}')
        s3.download_file(bucket, key, str(dest))


if __name__ == '__main__':
    main()

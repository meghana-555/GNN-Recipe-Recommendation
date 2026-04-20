import boto3, os
from botocore.client import Config
from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('CHAMELEON_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('CHAMELEON_SECRET_KEY'),
    endpoint_url='https://chi.tacc.chameleoncloud.org:7480',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1')
paginator = s3.get_paginator('list_objects_v2')
print('=== Objects under train/ ===')
for page in paginator.paginate(Bucket='ObjStore_proj14', Prefix='train/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        size = obj['Size']
        modified = obj['LastModified']
        print(f"  {key}  ({size} bytes)  {modified}")
print('\n=== All top-level prefixes ===')
resp = s3.list_objects_v2(Bucket='ObjStore_proj14', Delimiter='/')
for prefix in resp.get('CommonPrefixes', []):
    print(f"  {prefix['Prefix']}")

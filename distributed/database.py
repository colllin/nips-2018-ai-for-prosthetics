import os
import math
import uuid
import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pymongo import MongoClient

client = MongoClient(f'mongodb+srv://{os.environ["MONGODB_USERNAME"]}:{os.environ["MONGODB_PASSWORD"]}@cluster0-l6ac3.mongodb.net/ProstheticsEnv?retryWrites=true')
db = client.ProstheticsEnv

instance_id = str(uuid.uuid4())

# Mutates `obj`
def without_empty_vals(obj):
    copy = {}
    for k,v in obj.items():
        if getattr(v, 'items', False):
            copy[k] = without_empty_vals(v)
        elif v:
            copy[k] = v
    return copy

def persist_timesteps(df_timesteps, page_size=150):
    n_pages = math.ceil(len(df_timesteps) / page_size)
    for i in range(n_pages):
        start = i * page_size
        end = start + page_size
        records = df_timesteps.iloc[start:end].to_dict('records')
        for r in records:
            del r['info']
            r['obs'] = without_empty_vals(r['obs'])
        db.timesteps.insert_many(records, ordered=False)
    
def sample_timesteps(n=100):
    return list(db.timesteps.aggregate([
        {'$sample': {'size': n}},
    ]))

def get_total_timesteps():
    return db.timesteps.count_documents({})

def persist_event(event_type, body):
    # Add time and unique process ID
    db.events.insert_one({
        'type': event_type,
        'body': body,
        'instance_id': instance_id,
        'created_at': datetime.datetime.now().isoformat(),
    })




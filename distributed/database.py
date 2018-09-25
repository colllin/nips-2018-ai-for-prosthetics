import os
import math
import uuid
import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pymongo import MongoClient

clients = []
instance_id = str(uuid.uuid4())

def clear_clients_for_thread():
    while clients: clients.pop()

def get_client_threadsafe():
    if not clients:
        clients.append(
            MongoClient(
                host=os.environ["MONGODB_HOST"], 
                username=os.environ["MONGODB_USERNAME"],
                password=os.environ["MONGODB_PASSWORD"],
                retryWrites=True,
            )
        )
    return clients[0]

def get_db_threadsafe():
    return get_client_threadsafe()[os.environ['MONGODB_DATABASE']]


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
    db = get_db_threadsafe()
    n_pages = math.ceil(len(df_timesteps) / page_size)
    for i in range(n_pages):
        start = i * page_size
        end = start + page_size
        records = df_timesteps.iloc[start:end].to_dict('records')
        for r in records:
            del r['info']
            r['obs'] = without_empty_vals(r['obs'])
        db.timesteps.insert_many(records, ordered=False)
    

def sample_timesteps(n=100, n_obs_history=1):
    db = get_db_threadsafe()
    docs = list(db.timesteps.aggregate([
        {
            '$sample': {
                'size': round(n*1.2)
            }
        }, {
            '$match': {
                'i_step': {
                    '$ne': 0
                }
            }
        }, {
            '$limit': n
        }
    ]))
    past_timesteps = db.timesteps.find(
        filter={
            '$or': [{
                'episode_uuid': d['episode_uuid'],
                'i_step': {
                    '$in': [d['i_step']-h for h in range(1, 1+n_obs_history)],
                },
            } for d in docs],
        }, 
        projection={
            '_id': False, 
            'episode_uuid': True, 
            'i_step': True, 
            'obs': True
        }
    )
    docs_by_episode = {}
    for d in docs:
        if not d['episode_uuid'] in docs_by_episode:
            docs_by_episode[d['episode_uuid']] = []
        docs_by_episode[d['episode_uuid']].append(d)
    for pt in past_timesteps:
        for d in docs_by_episode[pt['episode_uuid']]:
            step_diff = d['i_step'] - pt['i_step']
            if step_diff < 0 or step_diff > n_obs_history: continue
            d[f'obs_t-{step_diff}'] = pt['obs']
    # import json
    # print(json.dumps(docs[0], indent=4))
    return docs

def get_total_timesteps():
    db = get_db_threadsafe()
    return db.timesteps.count_documents({})

def persist_event(event_type, body):
    db = get_db_threadsafe()
    # Add timestamp and unique process ID
    db.events.insert_one({
        'type': event_type,
        'body': body,
        'instance_id': instance_id,
        'created_at': datetime.datetime.now().isoformat(),
    })





from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from pymongo import MongoClient

client = MongoClient(f'mongodb+srv://{process.env.MONGODB_USERNAME}:{process.env.MONGODB_PASSWORD@cluster0-l6ac3.mongodb.net/ProstheticsEnv?retryWrites=true')
db = client.ProstheticsEnv

def persist_timesteps(df_timesteps):
    db.timesteps.insert_many(df_timesteps.to_dict(), ordered=False)
    
def sample_timesteps(n=100):
    return list(db.timesteps.aggregate([
        {'$sample': {'size': n}},
    ]))


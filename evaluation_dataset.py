from peewee import IntegerField, TextField, BlobField, FloatField, Model, SqliteDatabase
import base64
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
import pytorch_lightning as pl
from random import randrange

# DB_PATH = '2021-07-31-lichess-evaluations-37MM.db'

class Evaluations(Model):
  id = IntegerField()
  fen = TextField()
  eval = FloatField()
  ego_binary = BlobField()
  alter_binary = BlobField()

  class Meta:
    database = None

  def binary_base64(self):
    return base64.b64encode(self.binary)

# # Initialize database
# def initialize_db():
#     db = SqliteDatabase(DB_PATH)
#     Evaluations._meta.database = db  # Attach database to model class
#     db.connect()
#     return db

# db = initialize_db()

# Dataset
class EvaluationDataset(IterableDataset):
    def __init__(self, count, db_path):
        self.count = count
        self.db_path = db_path

    def __iter__(self):
        return self

    def __next__(self):
        idx = randrange(self.count)
        return self.__getitem__(idx)

    def __len__(self):
        return self.count

    def initialize_db(self):
      db = SqliteDatabase(self.db_path)
      Evaluations._meta.database = db  # Attach database to model class
      db.connect()

    def __getitem__(self, idx):
        # entry = Evaluations.get(Evaluations.id == idx+1)
        # id, fen, eval, ego_bin, alter_bin = entry.id, entry.fen, entry.eval, entry.ego_binary, entry.alter_binary
        evaluation = Evaluations.get(Evaluations.id == idx + 1)
        ego_bin = np.frombuffer(evaluation.ego_binary, dtype=np.uint8)
        # ego_bin = np.unpackbits(ego_bin, axis=0).astype(np.float32)
        alter_bin = np.frombuffer(evaluation.alter_binary, dtype=np.uint8)
        # alter_bin = np.unpackbits(alter_bin, axis=0).astype(np.float32)
        
        evaluation_value = self.normalize_evaluation(evaluation.eval)
        eval_data = np.array([evaluation_value]).astype(np.single)
        
        return {'ego_bin': ego_bin, 'alter_bin': alter_bin, 'eval': eval_data}

    def normalize_evaluation(self, eval_value):
        return np.clip(eval_value, -15, 15)

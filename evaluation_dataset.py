# # Database interactions
# from peewee import SqliteDatabase, Model, IntegerField, TextField, BlobField, FloatField

# # '2021-07-31-lichess-evaluations-37MM.db'

# class EvaluationDatabase:
#     def __init__(self, db_path):
#         self.db = SqliteDatabase(db_path)

#         class Evaluations(Model):
#             id = IntegerField()
#             fen = TextField()
#             binary = BlobField()
#             eval = FloatField()

#             class Meta:
#                 database = self.db

#         self.Evaluations = Evaluations

#     def get_evaluation(self, idx):
#         return self.Evaluations.get(self.Evaluations.id == idx+1)

# # Dataset
# import torch
# import numpy as np
# from torch.utils.data import IterableDataset
# from random import randrange

# class EvaluationDataset(IterableDataset):
#     def __init__(self, count, db_path):
#         self.count = count
#         self.db_path = db_path
#         self.db = None

#     def __iter__(self):
#         return self

#     def __next__(self):
#         idx = randrange(self.count)
#         return self[idx]

#     def __len__(self):
#         return self.count

#     def __getitem__(self, idx):
#         if self.db is None:
#             self.db = EvaluationDatabase(self.db_path)
#             self.db.db.connect()
#         eval_db = self.db.get_evaluation(idx)
#         bin_array = np.frombuffer(eval_db.binary, dtype=np.uint8)
#         bin_array = np.unpackbits(bin_array, axis=0).astype(np.single)
#         eval_value = max(min(eval_db.eval, 15), -15)
#         eval_value = np.array([eval_value]).astype(np.single)
#         return {'binary': bin_array, 'eval': eval_value}
    
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
    binary = BlobField()
    eval = FloatField()

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

    def initalize_db(self):
      db = SqliteDatabase(self.db_path)
      Evaluations._meta.database = db  # Attach database to model class
      db.connect()

    def __getitem__(self, idx):
        evaluation = Evaluations.get(Evaluations.id == idx + 1)
        binary_data = np.frombuffer(evaluation.binary, dtype=np.uint8)
        binary_data = np.unpackbits(binary_data, axis=0).astype(np.single)
        
        evaluation_value = self.normalize_evaluation(evaluation.eval)
        eval_data = np.array([evaluation_value]).astype(np.single)
        
        return {'binary': binary_data, 'eval': eval_data}

    def normalize_evaluation(self, eval_value):
        return np.clip(eval_value, -15, 15)

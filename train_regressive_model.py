import time
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from evaluation_model import EvaluationModelHalfKP
from evaluation_dataset import EvaluationDataset# , initialize_db
from datetime import datetime

# DB_PATH = '2021-07-31-lichess-evaluations-37MM.db'
# LABEL_COUNT = 37164639  # This should be provided somewhere in your code or as a constant
DB_PATH = 'new_database.db'
LABEL_COUNT = 50000  # This should be provided somewhere in your code or as a constant

# Load the EvaluationModel and EvaluationDataset
dataset = EvaluationDataset(count=LABEL_COUNT, db_path=DB_PATH)

def worker_init_fn(worker_id):
    dataset.initialize_db()

configs = [
    {"arch": [1,1], "batch_size": 1024},
    # {"layer_count": 6, "batch_size": 1024},
]

if __name__ == '__main__':
  # Train
  for config in configs:
      current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
      version_name = f'{current_time}-batch_size-{config["batch_size"]}-arch-{"-".join(map(str, config["arch"]))}'
      logger = pl.loggers.TensorBoardLogger("lightning_logs", name="chessml", version=version_name)
      trainer = pl.Trainer(gpus=1, precision=16, max_epochs=5, auto_lr_find=True, logger=logger, accelerator='mps')
      model = EvaluationModelHalfKP(arch=config["arch"], batch_size=config["batch_size"], learning_rate=1e-3)
      train_loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
      trainer.fit(model, train_loader)
      torch.save(model.state_dict(), "models/" + version_name + "_model.pth")

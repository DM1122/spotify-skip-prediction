"""Train RNN"""
# stdlib
import logging
from pathlib import Path

# external
import torch
import torchinfo

# project
import gym
import models
import rnn_data_loader

# region paths config
log_path = Path("logs/scripts")
output_path = Path("output")
# endregion

# region logging config
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=(log_path / Path(__file__).stem).with_suffix(".log"),
    filemode="w",
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
LOG = logging.getLogger(__name__)
# endregion

# region main
LOG.info("Starting RNN training")
device = gym.get_device()
LOG.info(f"Using {device}")

# dataloaders
dataloader_train = rnn_data_loader.read_rnn_dataloaders('encoded_features', 'labels', 'train', iterator='1')
dataloader_test= rnn_data_loader.read_rnn_dataloaders('encoded_features', 'labels', 'test', iterator='1')
dataloader_valid= rnn_data_loader.read_rnn_dataloaders('encoded_features', 'labels', 'valid', iterator='1')

# model definiton
LOG.info("Instantiating model")
model = models.RNN(input_size=4, hidden_size=64, num_rnn_layers=1, output_size=1).to(
    device
)
model=model.double()
summary = torchinfo.summary(
    model=model,
    input_data=next(iter(dataloader_train))[0],
    col_names=("input_size", "output_size", "num_params"),
    verbose=0,
)
LOG.info(f"Model:\n{summary}")

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss(reduction="sum") # <-------------------------- TODO: change from torch.nn.MSELoss() to torch.nn.BCEWithLogitsLoss() @Eddie

# training
trainer = gym.Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    logname="test_time_series",
)
tb = trainer.train(iterations=500)
tb.close()

loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
LOG.info(f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%")

trainer.save_model("rnn_spotify")

# endregion

"""Train RNN"""
# stdlib
import logging
from pathlib import Path

# external
import torch
import torchinfo

# project
from spotify_skip_prediction.core import gym, models
from spotify_skip_prediction.datahandler import data_handler

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
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)
# endregion

# region main
LOG.info("Starting RNN training")
device = gym.get_device()
LOG.info(f"Using {device}")

# dataloaders
dataloader_train, dataloader_test, dataloader_valid = data_handler.get_dataloaders()

# model definiton
LOG.info("Instantiating model")
model = models.RNN(input_size=5, hidden_size=16, num_rnn_layers=4, output_size=1).to(
    device
)
summary = torchinfo.summary(
    model=model,
    input_data=next(iter(dataloader_train))[0],
    col_names=("input_size", "output_size", "num_params"),
    verbose=0,
)
LOG.info(f"Model:\n{summary}")

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.03)
criterion = torch.nn.MSELoss(reduction="sum")

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
tb = trainer.train(iterations=100)
tb.close()

loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
LOG.info(f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%")

trainer.save_model("rnn_spotify")

# endregion

"""Use trained autoencoder model to compress dataset."""
# stdlib
import logging
from pathlib import Path

# external
import torch

# project
from spotify_skip_prediction.core import gym
from spotify_skip_prediction.datahandler import autoencoder_data_loaders

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
LOG.info("Starting dataset compression")
device = gym.get_device()
LOG.info(f"Using {device}")

# train
LOG.info("Compressing train dataset")
(
    dataloader_train,
    dataloader_test,
    dataloader_valid,
) = autoencoder_data_loaders.read_autoencoder_dataloaders(batch_size=134303) # process the whole thing


model = torch.load("models/autoencoder_spotify.pt")

inputs_train, labels_train = next(iter(dataloader_train))
inputs_train, labels_train = inputs_train.to(device), labels_train.to(device)

# train loss
with torch.no_grad():
    logits_train = model(inputs_train)
    

# endregion

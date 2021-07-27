"""Use trained autoencoder model to compress dataset."""
# stdlib
import logging
from pathlib import Path

# external
import torch

# project
from spotify_skip_prediction.core import gym
from spotify_skip_prediction.datahandler import data_loaders

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


dataloader = data_loaders.get_autoencoder_dataloaders_no_split(batch_size=128)


model = torch.load("models/autoencoder_spotify.pt")

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass
        logits = model.encoder(inputs)

        # save tensor or send to datahandler helper to store dataset


# endregion

"""Use trained autoencoder model to compress dataset."""
# stdlib
import logging
from pathlib import Path

# external
import torch

# project
from spotify_skip_prediction.core import gym
from spotify_skip_prediction.datahandler import autoencoder_data_loaders, rnn_data_loader

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


(
    dataloader_train,
    dataloader_test,
    dataloader_valid,
) = autoencoder_data_loaders.read_autoencoder_dataloaders(batch_size=134303) # process the whole thing at once

model = torch.load("models/autoencoder_spotify_final.pt")

#region train dataset
LOG.info("Compressing train dataset")

inputs_train, labels_train = next(iter(dataloader_train))
inputs_train, labels_train = inputs_train.to(device), labels_train.to(device)

with torch.no_grad():
    logits_train = model(inputs_train)

rnn_data_loader.get_rnn_dataloaders(encoded_data=logits_train, dataset_type="train")
#endregion

#region test dataset
LOG.info("Compressing test dataset")

inputs_test, labels_test = next(iter(dataloader_test))
inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

with torch.no_grad():
    logits_test = model(inputs_test)

rnn_data_loader.get_rnn_dataloaders(encoded_data=logits_test, dataset_type="test")
#endregion

#region valid dataset
LOG.info("Compressing valid dataset")

inputs_valid, labels_valid = next(iter(dataloader_valid))
inputs_valid, labels_valid = inputs_valid.to(device), labels_valid.to(device)

with torch.no_grad():
    logits_valid = model(inputs_valid)

rnn_data_loader.get_rnn_dataloaders(encoded_data=logits_valid, dataset_type="valid")
#endregion
LOG.INFO("Done")

# endregion

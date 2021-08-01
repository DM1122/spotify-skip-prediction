"""Use trained autoencoder model to compress dataset."""
# stdlib
import logging
from pathlib import Path

# external
import torch

# project
import gym
import autoencoder_data_loaders
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
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)
# endregion


# region main
LOG.info("Starting dataset compression")
device = gym.get_device()
LOG.info(f"Using {device}")


dataloader_train, dataloader_valid, dataloader_test = autoencoder_data_loaders.read_autoencoder_dataloaders(iterator="1", batch_size=128)


model = torch.load("models/autoencoder_spotify.pt")

logits_out=torch.empty((0, 4), dtype=torch.float).to(device)
with torch.no_grad():
    for inputs, labels in dataloader_train:
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass
        logits = model.encoder(inputs)
        logits_out=torch.cat((logits_out, logits))
        # save tensor or send to datahandler helper to store dataset
logits_out=torch.cat((logits_out, logits_out[-1].unsqueeze(0))) #somehow lost a tensor so just copy the last one and fudge it a bit
rnn_data_loader.get_rnn_dataloaders(logits_out, 'train', iterator='1')



logits_out=torch.empty((0, 4), dtype=torch.float).to(device)
with torch.no_grad():
    for inputs, labels in dataloader_valid:
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass
        logits = model.encoder(inputs)

        logits_out=torch.cat((logits_out, logits))
        # save tensor or send to datahandler helper to store dataset
logits_out=torch.cat((logits_out, logits_out[-1].unsqueeze(0))) #somehow lost a tensor so just copy the last one and fudge it a bit
rnn_data_loader.get_rnn_dataloaders(logits_out, 'valid', iterator='1')


logits_out=torch.empty((0, 4), dtype=torch.float).to(device)
with torch.no_grad():
    for inputs, labels in dataloader_test:
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass
        logits = model.encoder(inputs)

        logits_out=torch.cat((logits_out, logits))
        # save tensor or send to datahandler helper to store dataset
logits_out=torch.cat((logits_out, logits_out[-1].unsqueeze(0))) #somehow lost a tensor so just copy the last one and fudge it a bit
rnn_data_loader.get_rnn_dataloaders(logits_out, 'test', iterator='1')
# endregion

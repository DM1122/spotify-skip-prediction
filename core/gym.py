"""A gym for getting those models in shape."""

# stdlib
from datetime import datetime

# external
import torch
import torchvision
from torch.utils import tensorboard


class Trainer:
    """Trainer class for training models.

    Args:
        model (torch.nn.Module): Pytorch model.
        dataloader_train (torch.utils.data.Dataloader): Dataloader for train data.
        dataloader_test (torch.utils.data.Dataloader): Dataloader for test data.
        optimizer (torch.optim.Optimizer): Training optimizer.
        criterion (torch.nn._Loss): Loss function. Must be configured with
            reduction mode in "sum".
        device (torch.device): Device on which to train model.
        logname (str): Name of directory in which to log tensorboard runs.

    Attributes:
        tb (SummaryWriter): Used to log info to Tensorboard.
    """

    def __init__(
        self,
        model,
        dataloader_train,
        dataloader_test,
        optimizer,
        criterion,
        device,
        logname=None,
    ):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logname = logname

        if self.logname is not None:
            logdir = (
                "logs/tensorboard/"
                + f"{self.logname}/"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            logdir = "logs/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb = SummaryWriter(logdir)

        # log model graph
        self.tb.add_graph(
            model=self.model, input_to_model=next(iter(self.dataloader_train))[0]
        )

    def train(self, iterations):
        """Train the model.

        Args:
            iterations (int): Number of training iterations. Each iteration is composed
                of a forward and backwards pass.
        """
        i = 0
        epoch = 1

        self.__checkpoint(step=i)

        while i < iterations:
            print(f"Epoch {epoch} commencing")
            for inputs, labels in self.dataloader_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                logits = self.model(inputs)
                loss_batch = self.criterion(logits, labels)
                loss = loss_batch / self.dataloader_train.batch_size

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1

                if i % 10 == 0:
                    self.__checkpoint(step=i, inputs=inputs, labels=labels)

                if i >= iterations:
                    self.__checkpoint(
                        step=i, inputs=inputs, labels=labels, log_hparam=True
                    )
                    break

            epoch += 1

        print("Done!")
        self.tb.close()

    def test(self, dataloader):
        """Computes test loss and accuracy metrics using a given dataloader

        Returns:
            (tuple): tuple containing:
                loss (float): Model loss on test dataset
                acc (float): Model fractional accuracy on test dataset.
        """
        loss_sum = 0
        correct_sum = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                logits = self.model(inputs)
                loss_batch = self.criterion(logits, labels)

                # metrics
                loss_sum += loss_batch
                predictions = torch.argmax(input=logits, dim=1, keepdim=False)
                try:  # only computes accuracy for classification tasks
                    correct_sum += (
                        (predictions == labels).type(torch.float).sum().item()
                    )
                except:
                    correct_sum = 0

        loss = loss_sum / len(self.dataloader_test)
        acc = correct_sum / len(self.dataloader_test)

        return loss, acc

    def __checkpoint(self, step, inputs=None, labels=None, log_hparam=False):
        """Conducts a checkpoint of the current model state by testing and logging to
        Tensorboard.

        Args:
            step (int): The current training iteration.
            inputs (torch.Tensor, optional): Inputs of current batch. Defaults to None.
                If None, will sample a batch from dataloader. Useful when checkpointing
                before first forward pass is done.
            labels (torch.Tensor, optional): Labels of current batch. Defaults to None.
                If None, will sample a batch from dataloader. Useful when checkpointing
                before first forward pass is done.
            log_hparam (bool, optional): Logs the hyperameters and final test metrics
                for the run. Should be set to True at the end of model training, but not
                before. Defaults to False.
        """
        if inputs is None or labels is None:  # checkpoint running at iteration 0
            inputs, labels = next(iter(self.dataloader_train))
            inputs, labels = inputs.to(self.device), labels.to(self.device)

        # train loss
        with torch.no_grad():
            logits = self.model(inputs)
            loss_batch = self.criterion(logits, labels)

        loss_train = loss_batch / self.dataloader_train.batch_size

        # train accuracy
        predictions = torch.argmax(input=logits, dim=1, keepdim=False)
        try:  # only computes accuracy for classification tasks
            correct_sum = (predictions == labels).type(torch.float).sum().item()
        except:
            correct_sum = 0
        acc_train = correct_sum / self.dataloader_train.batch_size

        # test metrics
        loss_test, acc_test = self.test(dataloader=self.dataloader_test)

        # log metrics
        self.tb.add_scalar(
            tag="Loss/Train",
            scalar_value=loss_train,
            global_step=step,
            new_style=False,
        )
        self.tb.add_scalar(
            tag="Loss/Test",
            scalar_value=loss_test,
            global_step=step,
            new_style=False,
        )
        self.tb.add_scalar(
            tag="Accuracy/Train",
            scalar_value=acc_train,
            global_step=step,
            new_style=False,
        )
        self.tb.add_scalar(
            tag="Accuracy/Test",
            scalar_value=acc_test,
            global_step=step,
            new_style=False,
        )

        # log params
        for layer_name, param in self.model.named_parameters():
            self.tb.add_histogram(
                tag=self.logname + "." + layer_name, values=param, global_step=step
            )

        # log samples
        img_grid = torchvision.utils.make_grid(inputs)
        self.tb.add_image(tag="Input", img_tensor=img_grid, global_step=step)

        img_grid = torchvision.utils.make_grid(logits)
        self.tb.add_image(tag="Output", img_tensor=img_grid, global_step=step)

        # log hparams
        if log_hparam is True:
            self.tb.add_hparams(
                hparam_dict={
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "bs": self.dataloader_train.batch_size,
                },
                metric_dict={"Metric/loss": loss_test, "Metric/acc": acc_test},
            )

        # display
        print()
        print(f"Iteration {step} stats:")
        print(f"Examples seen:\t{step*self.dataloader_train.batch_size}")
        print(f"Train loss:\t{loss_train:.3f}\tTest loss:\t{loss_test:.3f}")
        print(f"Train acc:\t{acc_train*100:.2f}%\tTest acc:\t{acc_test*100:.2f}%")


class SummaryWriter(tensorboard.SummaryWriter):
    """A modified Tensorboard SummaryWriter class to address the issue of the sucky
    directory handling when using the hparam api. See:
    https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if isinstance(hparam_dict) is not dict or isinstance(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = torch.utils.tensorboard.summary.hparams(
            hparam_dict, metric_dict
        )

        logdir = self._get_file_writer().get_logdir()

        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def get_device():
    """Gets the available device for training. Either CPU or cuda GPU.

    Returns:
        torch.device: The device.
    """
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    return device

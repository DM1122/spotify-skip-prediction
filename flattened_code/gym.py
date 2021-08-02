"""A gym for getting those models in shape."""

# stdlib
import logging
import os
import winsound
from datetime import datetime

# external
import sklearn
import skopt
import torch
import torchvision
from pmdarima import datasets as pmd
from sklearn import datasets as skd
from torch.utils import tensorboard
from tqdm.auto import tqdm

# project
import models
import autoencoder_data_loaders
import rnn_data_loader
import datalib, plotlib

LOG = logging.getLogger(__name__)


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
        writer (SummaryWriter): Used to log info to Tensorboard.
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
        self.writer = SummaryWriter(logdir)

        # log model graph
        self.writer.add_graph(
            model=self.model, input_to_model=next(iter(self.dataloader_train))[0]
        )

    def train(self, iterations):
        """Train the model.

        Args:
            iterations (int): Number of training iterations. Each iteration is composed
                of a forward and backwards pass.
        Returns:
            SummaryWriter: The tensorboard SummaryWriter object for the run. Used by
            Tuner to log hyperparameters after run. Must call SummaryWriter.close()
            after usage.
        """
        i = 0
        epoch = 1

        LOG.info("Beginning training session")
        self._checkpoint(step=i)

        pbar = tqdm(
            desc="Training model",
            total=iterations,
            leave=True,
            unit="it",
            dynamic_ncols=True,
        )
        while i < iterations:
            LOG.debug("Epoch {epoch} commencing")
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
                    self._checkpoint(step=i, inputs=inputs, labels=labels)

                if i >= iterations:
                    self._checkpoint(step=i, inputs=inputs, labels=labels)
                    break
                pbar.update()

            epoch += 1
        pbar.close()

        return self.writer

    def test(self, dataloader, samples=None):
        """Computes test loss and accuracy metrics using a given dataloader

        Args:
            dataloader: Pytorch dataloader.
            samples (int): Number of samples to test on from dataloader.

        Returns:
            (tuple): tuple containing:
                loss (float): Model loss on test dataset
                acc (float): Model fractional accuracy on test dataset.
        """
        loss_sum = 0
        correct_sum = 0

        samples = len(dataloader) if samples is None else samples
        i = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                logits = self.model(inputs)
                loss_batch = self.criterion(logits, labels)

                # metrics
                loss_sum += loss_batch
                predictions = torch.argmax(input=logits, dim=1, keepdim=False)
                try:  # only compute accuracy for classification tasks
                    correct_sum += (
                        (predictions == labels).type(torch.float).sum().item()
                    )
                except RuntimeError:
                    correct_sum = 0

                i += 1
                if i >= samples:
                    break

        loss = loss_sum / len(dataloader)
        acc = correct_sum / len(dataloader)

        return loss, acc

    def _checkpoint(self, step, inputs=None, labels=None):
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
        try:  # only compute accuracy for classification tasks
            correct_sum = (predictions == labels).type(torch.float).sum().item()
        except RuntimeError:
            correct_sum = 0
        acc_train = correct_sum / self.dataloader_train.batch_size

        # test metrics
        loss_test, acc_test = self.test(dataloader=self.dataloader_test)

        # log metrics
        self.writer.add_scalar(
            tag="Loss/Train",
            scalar_value=loss_train,
            global_step=step,
            new_style=False,
        )
        self.writer.add_scalar(
            tag="Loss/Test",
            scalar_value=loss_test,
            global_step=step,
            new_style=False,
        )
        self.writer.add_scalar(
            tag="Accuracy/Train",
            scalar_value=acc_train,
            global_step=step,
            new_style=False,
        )
        self.writer.add_scalar(
            tag="Accuracy/Test",
            scalar_value=acc_test,
            global_step=step,
            new_style=False,
        )

        # log params
        for layer_name, param in self.model.named_parameters():
            self.writer.add_histogram(
                tag=self.logname + "." + layer_name, values=param, global_step=step
            )

        # log samples
        img_grid = torchvision.utils.make_grid(inputs)
        try:
            self.writer.add_image(tag="Input", img_tensor=img_grid, global_step=step)
        except TypeError:
            LOG.warning("Cannnot write image for given inputs shape")

        img_grid = torchvision.utils.make_grid(logits)
        try:
            self.writer.add_image(tag="Output", img_tensor=img_grid, global_step=step)
        except TypeError:
            LOG.warning("Cannnot write image for given logits shape")

        LOG.debug(
            (
                f"Iteration {step} stats:\n"
                f"Samples seen:\t{step*self.dataloader_train.batch_size}\n"
                f"Train loss:\t{loss_train:.3f}\tTest loss:\t{loss_test:.3f}\n"
                f"Train acc:\t{acc_train*100:.2f}%\tTest acc:\t{acc_test*100:.2f}%"
            )
        )

    def save_model(self, name):
        """Saves the current model to disk. To restore a model, call torch.load().

        Args:
            name (str): Model filename to save it as.
        """
        if not os.path.exists("models"):
            os.makedirs("models")

        LOG.info("Saving model to disk")
        torch.save(self.model, f"models/{name}.pt")


class Tuner:
    """A tuner for hyperparameter optimization. All model-specific tuners must subclass
    from this class."""

    def __init__(self):
        self.device = get_device()

        # the following must be defined in derived classes:
        self.space = None

    def tune(self, n_calls):
        """Tunes the defined black box function according to the defined hyperparamter
        space.

        Args:
            n_calls (int): Number of calls to black box function. Must be greater than
                or equal to 3.
        """

        opt_res = skopt.gp_minimize(
            func=self._blackbox,
            dimensions=self.space,
            base_estimator=None,
            n_calls=n_calls,
            n_initial_points=3,
            initial_point_generator="random",
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            x0=None,
            y0=None,
            random_state=None,
            verbose=True,
            callback=self._callback,
            n_points=10000,
            n_restarts_optimizer=None,
            xi=0.01,
            kappa=1.96,
            noise="gaussian",
            n_jobs=None,
            model_queue_size=None,
        )

        LOG.info(
            (
                "Tuning results:\n"
                f"Location of min:\t{opt_res.x}\n"
                f"Function value at min:\t{opt_res.fun}"
            )
        )

        # region plots
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plotlib.plot_skopt_evaluations(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_objective(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_convergence(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_regret(opt_res, f"logs/tuner/{time}/")
        # endregion

        winsound.MessageBeep()

    def explore(self, n_calls):
        """Conducts random search by uniform sampling within the given bounds for the
        defined black box function and hyperparamter space.

        Args:
            n_calls (int): Number of calls to black box function.
        """

        opt_res = skopt.gp_minimize(
            func=self._blackbox,
            dimensions=self.space,
            base_estimator=None,
            n_calls=n_calls,
            n_initial_points=n_calls,
            initial_point_generator="random",
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            x0=None,
            y0=None,
            random_state=None,
            verbose=True,
            callback=self._callback,
            n_points=10000,
            n_restarts_optimizer=None,
            xi=0.01,
            kappa=1.96,
            noise="gaussian",
            n_jobs=None,
            model_queue_size=None,
        )

        LOG.info(
            (
                "Exploration results:\n"
                f"Location of min:\t{opt_res.x}\n"
                f"Function value at min:\t{opt_res.fun}"
            )
        )

        # region plots
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        plotlib.plot_skopt_evaluations(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_objective(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_convergence(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_regret(opt_res, f"logs/tuner/{time}/")
        # endregion

        winsound.MessageBeep()

    def _blackbox(self, params):
        """The blackbox function to be optimized.

        Args:
            params (list): A list of hyperparameters to be evaluated at in the current
                call to the blackbox function.

        Returns:
            float: Function loss.
        """
        dataloader_train, dataloader_test, _ = self._get_dataloaders(params)
        model, optimizer, criterion = self._build_model(params)

        trainer = Trainer(
            model=model,
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            logname="RNN_test", # adjust log name
        )
        writer = trainer.train(iterations=100) #CHANGED

        loss, acc = trainer.test(dataloader=dataloader_test)

        self._record_hparams(writer, params, loss, acc)
        writer.close()

        return float(loss)

    @staticmethod
    def _callback(opt_res):
        LOG.debug(f"Hyperparameters that were just tested: {opt_res.x_iters[-1]}")

    def _get_dataloaders(self, params):
        raise NotImplementedError("Method must be implemented in derived classes.")

    def _build_model(self, params):
        raise NotImplementedError("Method must be implemented in derived classes.")

    def _record_hparams(self, writer, params, loss, acc):
        raise NotImplementedError("Method must be implemented in derived classes.")


class Tuner_Autoencoder_Test(Tuner):
    """A tuner for the autoencoder model."""

    def __init__(self):
        super().__init__()

        self.space = [
            skopt.space.space.Real(
                low=0.0001,
                high=1.0,
                name="lr",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="bs",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8], transform="identity", name="radius"
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8], transform="identity", name="embed_size"
            ),
        ]

    def _get_dataloaders(self, params):
        """Creates train, test, and validation dataloader."""
        # region dataset preprocessing
        features, _ = skd.load_wine(return_X_y=True)

        scaler = sklearn.preprocessing.StandardScaler()
        features = scaler.fit_transform(X=features)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float),
            torch.tensor(features, dtype=torch.float),
        )
        dataset_train, dataset_test, dataset_valid = torch.utils.data.random_split(
            dataset=dataset, lengths=(100, 39, 39)
        )
        # endregion

        # region dataloaders
        if get_device() == torch.device("cuda:0"):
            pin=False
        else:
            pin=True
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=int(params[1]),
            shuffle=True,
            pin_memory=pin,
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=True,
            pin_memory=pin,
        )
        dataloader_valid = torch.utils.data.DataLoader(
            dataset=dataset_valid,
            batch_size=1,
            shuffle=True,
            pin_memory=pin,
        )
        # endregion

        return dataloader_train, dataloader_test, dataloader_valid

    def _build_model(self, params):
        model = models.AutoEncoder(
            input_size=13, embed_size=int(params[3]), radius=int(params[2])
        ).to(self.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=float(params[0]))
        criterion = torch.nn.MSELoss(reduction="sum")

        return model, optimizer, criterion

    def _record_hparams(self, writer, params, loss, acc):
        """Records the current hyperparameters along with metrics to the Tensorboard
            writer.

        Args:
            writer (SummaryWriter): The tensorboard writer provided by the Trainer.
            params (list): list of evaluated hyperparameters.
            loss (torch.Tensor): Final test loss output from Trainer.
            acc (torch.Tensor): Final accuracy output from Trainer.
        """
        writer.add_hparams(
            hparam_dict={
                "lr": float(params[0]),
                "bs": int(params[1]),
                "radius": int(params[2]),
            },
            metric_dict={"Metric/loss": float(loss), "Metric/acc": float(acc)},
        )


class Tuner_Autoencoder_Spotify(Tuner):
    """A tuner for the Spotify autoencoder model."""

    def __init__(self):
        super().__init__()

        self.space = [
            skopt.space.space.Real(
                low=0.0001,
                high=1.0,
                name="lr",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="bs",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8], transform="identity", name="radius"
            ),
        ]

    def _get_dataloaders(self, params):
        (
            dataloader_train,
            dataloader_test,
            dataloader_valid,
        ) = autoencoder_data_loaders.read_autoencoder_dataloaders(iterator= '1', batch_size=int(params[1]))

        return dataloader_train, dataloader_test, dataloader_valid

    def _build_model(self, params):
        model = models.AutoEncoder(
            input_size=28, embed_size=8, radius=int(params[2])
        ).to(self.device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=float(params[0]))
        criterion = torch.nn.MSELoss(reduction="sum")

        return model, optimizer, criterion

    def _record_hparams(self, writer, params, loss, acc):
        """Records the current hyperparameters along with metrics to the Tensorboard
            writer.

        Args:
            writer (SummaryWriter): The tensorboard writer provided by the Trainer.
            params (list): list of evaluated hyperparameters.
            loss (torch.Tensor): Final test loss output from Trainer.
            acc (torch.Tensor): Final accuracy output from Trainer.
        """
        writer.add_hparams(
            hparam_dict={
                "lr": float(params[0]),
                "bs": int(params[1]),
                "radius": int(params[2]),
            },
            metric_dict={"Metric/loss": float(loss), "Metric/acc": float(acc)},
        )


class Tuner_RNN_Test(Tuner):
    """A tuner for the RNN model."""

    def __init__(self):
        super().__init__()

        self.space = [
            skopt.space.space.Real(
                low=0.0001,
                high=1.0,
                name="lr",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="bs",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32], transform="identity", name="hs"
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8], transform="identity", name="layers"
            ),
        ]

    def _get_dataloaders(self, params):
        # region data preprocessing

        # import
        LOG.info("Importing data")
        data = pmd.stocks.load_msft()
        LOG.info(f"Data:\n{data}")

        # filter
        LOG.info("Filtering data")
        data = data[["Open", "High", "Low", "Close", "Volume"]]

        # create labels
        LOG.info("Creating labels")
        data["Close_Future"] = data["Close"].shift(-1)
        data = data[:-1]
        LOG.debug(f"Data preprocessed:\n{data}")

        # split
        LOG.info("Splitting data")
        data_train = data[0:6000]
        LOG.debug(f"Data train:\n{data_train}")
        data_test = data[6000:7000]
        LOG.debug(f"Data test:\n{data_test}")
        data_valid = data[7000:7983]
        LOG.debug(f"Data valid:\n{data_valid}")

        # features and labels
        LOG.info("Extracting features and labels")
        features_train = data_train.drop("Close_Future", axis=1)
        LOG.debug(f"Features train:\n{features_train}")
        labels_train = data_train.loc[:, ["Close_Future"]]
        LOG.debug(f"Labels train:\n{labels_train}")

        features_test = data_test.drop("Close_Future", axis=1)
        labels_test = data_test.loc[:, ["Close_Future"]]

        features_valid = data_valid.drop("Close_Future", axis=1)
        labels_valid = data_valid.loc[:, ["Close_Future"]]

        # scaling
        LOG.info("Scaling data")
        scaler_features = sklearn.preprocessing.StandardScaler(
            with_mean=False, with_std=True
        )
        scaler_labels = sklearn.preprocessing.StandardScaler(
            with_mean=False, with_std=True
        )

        scaler_features.fit(features_train)
        LOG.debug(
            f"Feature scaler stats:\n"
            f"Scale:\t{scaler_features.scale_}\nMean:\t{scaler_features.mean_}\n"
            f"Var:\t{scaler_features.var_}\n"
            f"Samples:\t{scaler_features.n_samples_seen_}\n"
        )
        scaler_labels.fit(labels_train)
        LOG.debug(
            f"Label scaler stats:\n"
            f"Scale:\t{scaler_labels.scale_}\nMean:\t{scaler_labels.mean_}\n"
            f"Var:\t{scaler_labels.var_}\nSamples:\t{scaler_labels.n_samples_seen_}\n"
        )

        features_train = scaler_features.transform(features_train)
        LOG.debug(f"Features train scaled {features_train.shape}:\n{features_train}")
        labels_train = scaler_labels.transform(labels_train)
        LOG.debug(f"Labels train scaled {labels_train.shape}:\n{labels_train}")

        features_test = scaler_features.transform(features_test)
        labels_test = scaler_labels.transform(labels_test)

        features_valid = scaler_features.transform(features_valid)
        labels_valid = scaler_labels.transform(labels_valid)

        # reshaping
        LOG.info("Reshaping features and labels")
        features_train = datalib.split_sequences(sequences=features_train, n_steps=7)
        LOG.debug(f"Features train reshaped {features_train.shape}:\n{features_train}")
        labels_train = datalib.split_sequences(sequences=labels_train, n_steps=7)
        LOG.debug(f"Labels train reshaped {labels_train.shape}:\n{labels_train}")

        features_test = datalib.split_sequences(sequences=features_test, n_steps=7)
        labels_test = datalib.split_sequences(sequences=labels_test, n_steps=7)

        features_valid = datalib.split_sequences(sequences=features_valid, n_steps=7)
        labels_valid = datalib.split_sequences(sequences=labels_valid, n_steps=7)
        # endregion

        # region datasets
        LOG.info("Creating datasets")
        features_train = torch.tensor(features_train, dtype=torch.float)
        labels_train = torch.tensor(labels_train, dtype=torch.float)

        features_test = torch.tensor(features_test, dtype=torch.float)
        labels_test = torch.tensor(labels_test, dtype=torch.float)

        features_valid = torch.tensor(features_valid, dtype=torch.float)
        labels_valid = torch.tensor(labels_valid, dtype=torch.float)

        dataset_train = torch.utils.data.TensorDataset(
            features_train,
            labels_train,
        )
        LOG.debug(f"Dataset train:\n{dataset_train}")

        dataset_test = torch.utils.data.TensorDataset(
            features_test,
            labels_test,
        )

        dataset_valid = torch.utils.data.TensorDataset(
            features_valid,
            labels_valid,
        )
        # endregion

        # region dataloaders
        LOG.info("Creating dataloaders")
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=int(params[1]),
            shuffle=True,
            pin_memory=False,
        )
        LOG.debug(f"Dataloader train:\n{dataloader_train}")
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
        )
        dataloader_valid = torch.utils.data.DataLoader(
            dataset=dataset_valid,
            batch_size=1,
            shuffle=True,
            pin_memory=False,
        )
        # endregion

        return dataloader_train, dataloader_test, dataloader_valid

    def _build_model(self, params):
        model = models.RNN(
            input_size=5,
            hidden_size=int(params[2]),
            num_rnn_layers=int(params[3]),
            output_size=1,
        ).to(self.device)
        model=model.double()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=float(params[0]))
        criterion = torch.nn.MSELoss(reduction="sum")

        return model, optimizer, criterion

    def _record_hparams(self, writer, params, loss, acc):
        """Records the current hyperparameters along with metrics to the Tensorboard
            writer.

        Args:
            writer (SummaryWriter): The tensorboard writer provided by the Trainer.
            params (list): list of evaluated hyperparameters.
            loss (torch.Tensor): Final test loss output from Trainer.
            acc (torch.Tensor): Final accuracy output from Trainer.
        """
        writer.add_hparams(
            hparam_dict={
                "lr": float(params[0]),
                "bs": int(params[1]),
                "hs": int(params[2]),
                "layers": int(params[3]),
            },
            metric_dict={"Metric/loss": float(loss), "Metric/acc": float(acc)},
        )


class Tuner_RNN_Spotify(Tuner):
    """A tuner for the RNN model."""

    def __init__(self):
        super().__init__()

        self.space = [
            skopt.space.space.Real(
                low=0.0001,
                high=1.0,
                name="lr",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8],
                #categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="bs",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32], transform="identity", name="hs"
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8], transform="identity", name="layers"
            ),
        ]

    def _get_dataloaders(self, params):
        dataloader_train = rnn_data_loader.read_rnn_dataloaders("encoded_features", "labels", "train", "1", batch_size=int(params[1]))
        dataloader_test = rnn_data_loader.read_rnn_dataloaders("encoded_features", "labels", "test", "1", batch_size=int(params[1]))
        dataloader_valid = rnn_data_loader.read_rnn_dataloaders("encoded_features", "labels", "valid", "1", batch_size=int(params[1]))

        return dataloader_train, dataloader_test, dataloader_valid

    def _build_model(self, params):
        model = models.RNN(
            input_size=4, #MODIFIED TO 4 FROM 8
            hidden_size=int(params[2]),
            num_rnn_layers=int(params[3]),
            output_size=1,
        ).to(self.device)
        model = model.double()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=float(params[0]))
        criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

        return model, optimizer, criterion

    def _record_hparams(self, writer, params, loss, acc):
        """Records the current hyperparameters along with metrics to the Tensorboard
            writer.

        Args:
            writer (SummaryWriter): The tensorboard writer provided by the Trainer.
            params (list): list of evaluated hyperparameters.
            loss (torch.Tensor): Final test loss output from Trainer.
            acc (torch.Tensor): Final accuracy output from Trainer.
        """
        writer.add_hparams(
            hparam_dict={
                "lr": float(params[0]),
                "bs": int(params[1]),
                "hs": int(params[2]),
                "layers": int(params[3]),
            },
            metric_dict={"Metric/loss": float(loss), "Metric/acc": float(acc)},
        )


class SummaryWriter(tensorboard.SummaryWriter):
    """A modified Tensorboard SummaryWriter class to address the issue of the sucky
    directory handling when using the hparam api. See:
    https://github.com/pytorch/pytorch/issues/32651#issuecomment-643791116
    """

    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if not isinstance(hparam_dict, dict) or not isinstance(metric_dict, dict):
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
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

# from torchviz import make_dot
import torch.optim as optim
import torch.nn.functional as F


# Define the network class
class DragonNet(nn.Module):
    # Initialize the network layers
    def __init__(self, p):
        """p=params"""
        super(DragonNet, self).__init__()
        torch.set_default_dtype(torch.float32)
        self.is_tarnet = p["is_tarnet"]

        # Representation
        self.commonlayers = nn.ModuleList()
        self.commonlayers.append(nn.Linear(p["input_dim"], p["neurons_per_layer"]))
        self.commonlayers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
        )
        self.commonlayers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layer"])
        )
        for l in self.commonlayers:
            nn.init.normal_(l.weight, mean=0, std=0.05)

        self.t_layer = nn.Linear(p["neurons_per_layer"], 1)

        # Hypothesis
        self.y0layers = nn.ModuleList()
        self.y0layers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layerYs"])
        )
        self.y0layers.append(
            nn.Linear(p["neurons_per_layerYs"], p["neurons_per_layerYs"])
        )
        self.y0layers.append(nn.Linear(p["neurons_per_layerYs"], 1))

        self.y1layers = nn.ModuleList()
        self.y1layers.append(
            nn.Linear(p["neurons_per_layer"], p["neurons_per_layerYs"])
        )
        self.y1layers.append(
            nn.Linear(p["neurons_per_layerYs"], p["neurons_per_layerYs"])
        )
        self.y1layers.append(nn.Linear(p["neurons_per_layerYs"], 1))

    # Define the forward pass
    def forward(self, x):

        # input = x.clone()
        for layer in self.commonlayers:
            x = torch.nn.ELU()(layer(x))

        if self.is_tarnet:
            t_pred = torch.nn.Sigmoid()(self.t_layer(torch.zeros_like(x)))
        else:
            t_pred = torch.nn.Sigmoid()(self.t_layer(x))

        y0 = torch.nn.ELU()(self.y0layers[0](x))
        y0 = torch.nn.ELU()(self.y0layers[1](y0))
        y0_pred = self.y0layers[2](y0)

        y1 = torch.nn.ELU()(self.y1layers[0](x))
        y1 = torch.nn.ELU()(self.y1layers[1](y1))
        y1_pred = self.y1layers[2](y1)

        out = torch.cat([y0_pred, y1_pred, t_pred], dim=1)

        return out

def dragonnet_loss(pred, t_true, y_true, tarnet_bin=1.0, is_tarnet=False):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    -------
    loss: torch.Tensor
    """
    y0_pred = pred[:, 0]
    y1_pred = pred[:, 1]
    t_pred = pred[:, 2]

    t_pred = (t_pred + 0.001) / 1.002
    loss_t = torch.sum(F.binary_cross_entropy_with_logits(t_pred, t_true))

    loss0 = torch.sum((1.0 - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    #loss0 = torch.sum((1.0 - t_true) * F.binary_cross_entropy_with_logits(y0_pred, y_true))
    #loss1 = torch.sum(t_true * F.binary_cross_entropy_with_logits(y1_pred, y_true))
    
    loss_y = loss0 + loss1

    if is_tarnet:
        tarnet_bin = 0

    loss = loss_y + tarnet_bin * loss_t
    return loss

class DragonNetTrainer:
    def __init__(
        self, train_X, train_t, train_y, dag_edges=None, params=None, is_tarnet=False
    ):

        self.is_tarnet = is_tarnet
        # Hyperparameters --
        params_default = {
            "neurons_per_layer": 200,
            "reg_l2": 0.01,
            "val_split": 0.2,
            "batch_size": 64,
            "epochs": 300,
            "learning_rate": 1e-5,
            "momentum": 0.9,
        }
        if params is not None:
            for key, value in params.items():
                params_default[key] = value

        params = params_default

        self.batch_size = params["batch_size"]
        self.num_epochs = params["epochs"]
        self.early_stopping_value = 40
        params["input_dim"] = train_X.shape[1]
        params["neurons_per_layerYs"] = int(params["neurons_per_layer"] / 2)
        params["is_tarnet"] = is_tarnet

        # ------------------
        self.device = "cpu"
        self.early_stopping = True
        self.print_every = 400

        self.dag_edges = dag_edges
        self.train_losses = []
        self.val_losses = []

        self.batched_X = (
            torch.from_numpy(self.array_to_batch(train_X, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_t = (
            torch.from_numpy(self.array_to_batch(train_t, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        self.batched_y = (
            torch.from_numpy(self.array_to_batch(train_y, self.batch_size))
            .to(torch.float32)
            .to(self.device)
        )
        train_ids = np.random.choice(
            range(len(self.batched_X)),
            int(len(self.batched_X) * (1 - params["val_split"])),
        )
        val_ids = [i for i in range(len(self.batched_X)) if i not in train_ids]
        self.batched_train_X = self.batched_X[train_ids]
        self.batched_train_t = self.batched_t[train_ids]
        self.batched_train_y = self.batched_y[train_ids]
        self.batched_val_X = self.batched_X[val_ids]
        self.batched_val_t = self.batched_t[val_ids]
        self.batched_val_y = self.batched_y[val_ids]

        self.model = DragonNet(p=params)
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.model.commonlayers.parameters(), "weight_decay": 0},
                {"params": self.model.t_layer.parameters(), "weight_decay": 0},
                {
                    "params": self.model.y0layers.parameters(),
                    "weight_decay": params["reg_l2"],
                },
                {
                    "params": self.model.y1layers.parameters(),
                    "weight_decay": params["reg_l2"],
                },
            ],
            lr=params["learning_rate"],
            momentum=params["momentum"],
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=5,
            verbose=False,
            # mode="auto",
            eps=1e-8,
            cooldown=0,
            min_lr=0,
        )

        self.criterion = dragonnet_loss

    def train(self):
        best_val_loss = float("inf")
        loss_counter = 0
        for epoch in range(1, self.num_epochs + 1):
            losses = []
            # i= 0
            # aa =  zip(
            #     self.batched_train_X, self.batched_train_t, self.batched_train_y
            # )
            # X_batch, t_batch, y_batch = next(aa)
            for X_batch, t_batch, y_batch in zip(
                self.batched_train_X, self.batched_train_t, self.batched_train_y
            ):
                # print(i)
                # i += 1
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                if torch.isnan(pred[:, 2]).any():
                    print("pred is nan")
                    break

                loss = self.criterion(pred, t_batch, y_batch, is_tarnet=self.is_tarnet)
                if torch.isnan(loss):
                    print("loss is nan")
                    break
                    # print(4 + "h")

                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()

            if torch.isnan(pred[:, 2]).any():
                print("pred is nan _ epoch loop")
                break
            if torch.isnan(loss):
                print("loss is nan _ epoch loop")
                break
            self.scheduler.step(np.mean(losses))
            self.train_losses.append(np.mean(losses))

            # Validation
            losses = []
            for X_batch, t_batch, y_batch in zip(
                self.batched_val_X, self.batched_val_t, self.batched_val_y
            ):
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)

                if torch.isnan(y_pred[:, 2]).any():
                    print("pred is nan")
                    break

                loss = self.criterion(
                    y_pred, t_batch, y_batch, is_tarnet=self.is_tarnet
                )
                losses.append(loss.item())

            if torch.isnan(pred[:, 2]).any():
                print("pred is nan _epoch loop")
                break

            self.val_losses.append(np.mean(losses))

            # Save best model
            val_loss = self.val_losses[-1]
            if val_loss < best_val_loss:
                torch.save(self.model.state_dict(), "best_model.pth")
                best_val_loss = val_loss

            # Early stopping
            if self.early_stopping and epoch > 1:
                if self.val_losses[-1] > best_val_loss:

                    loss_counter += 1
                    if loss_counter == self.early_stopping_value:
                        break

                else:
                    loss_counter = 0

            if epoch % self.print_every == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {self.train_losses[-1]}, , Val Loss: {self.val_losses[-1]}"
                )

        self.model.load_state_dict(torch.load("best_model.pth"))

    def predict(self, test_X):
        test_X = torch.from_numpy(test_X).to(torch.float32).to(self.device)
        y_test = self.model(test_X)
        return y_test.detach().numpy()

    @staticmethod
    def array_to_batch(data, batch_size):

        num_batches = np.floor(len(data) / batch_size)

        if len(data) % batch_size == 0:
            batches = np.array_split(data, num_batches)
        else:
            batches = np.array_split(data[: -(len(data) % batch_size)], num_batches)

        return np.array(batches)


class TARNetTrainer(DragonNetTrainer):
    def __init__(
        self, train_X, train_t, train_y, dag_edges=None, params=None, is_tarnet=True
    ):
        super().__init__(train_X, train_t, train_y, dag_edges, params, is_tarnet)


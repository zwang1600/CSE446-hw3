if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = LinearLayer(input_dim, output_dim).double()

    def forward(self, x):
        return self.linear(x)
    
    def _get_name(self):
        return "Linear Regression"


class HiddenSigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(HiddenSigmoid, self).__init__()
        self.hidden_layer = LinearLayer(input_dim, hidden_dim).double()
        self.sigmoid = SigmoidLayer().double()
        self.output_layer = LinearLayer(hidden_dim, output_dim).double()

    def forward(self, x):
        hidden_output = self.sigmoid(self.hidden_layer(x))
        output = self.output_layer(hidden_output)
        return output
    
    def _get_name(self):
        return "Hidden -> Sigmoid"


class HiddenReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(HiddenReLU, self).__init__()
        self.hidden_layer = LinearLayer(input_dim, hidden_dim).double()
        self.relu = ReLULayer().double()
        self.output_layer = LinearLayer(hidden_dim, output_dim).double()

    def forward(self, x):
        hidden_output = self.relu(self.hidden_layer(x))
        output = self.output_layer(hidden_output)
        return output
    
    def _get_name(self):
        return "Hidden -> ReLU"


class HiddenSigmoidHiddenReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim) -> None:
        super(HiddenSigmoidHiddenReLU, self).__init__()
        self.hidden_layer1 = LinearLayer(input_dim, hidden_dim1).double()
        self.sigmoid = SigmoidLayer().double()
        self.hidden_layer2 = LinearLayer(hidden_dim1, hidden_dim2).double()
        self.relu = ReLULayer().double()
        self.output_layer = LinearLayer(hidden_dim2, output_dim).double()

    def forward(self, x):
        hidden_output1 = self.sigmoid(self.hidden_layer1(x))
        hidden_output2 = self.relu(self.hidden_layer2(hidden_output1))
        output = self.output_layer(hidden_output2)
        return output
    
    def _get_name(self):
        return "Hidden -> Sigmoid -> Hidden -> ReLU"


class HiddenReLUHiddenSigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim) -> None:
        super(HiddenReLUHiddenSigmoid, self).__init__()
        self.hidden_layer1 = LinearLayer(input_dim, hidden_dim1).double()
        self.relu = ReLULayer().double()
        self.hidden_layer2 = LinearLayer(hidden_dim1, hidden_dim2).double()
        self.sigmoid = SigmoidLayer().double()
        self.output_layer = LinearLayer(hidden_dim2, output_dim).double()

    def forward(self, x):
        hidden_output1 = self.relu(self.hidden_layer1(x))
        hidden_output2 = self.sigmoid(self.hidden_layer2(hidden_output1))
        output = self.output_layer(hidden_output2)
        return output
    
    def _get_name(self):
        return "Hidden -> ReLU -> Hidden -> Sigmoid"


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    training_history = {}
    models = [
        LinearRegression(2, 1),
        HiddenSigmoid(2, 2, 1),
        HiddenReLU(2, 2, 1),
        HiddenSigmoidHiddenReLU(2, 2, 2, 1),
        HiddenReLUHiddenSigmoid(2, 2, 2, 1)
    ]
    train_loader = DataLoader(dataset_train)
    val_loader = DataLoader(dataset_val)
    criterion = MSELossLayer()
    epochs = 2

    for model in models:
        optimizer = SGDOptimizer(model.parameters(), lr=1e-5)
        result = train(train_loader, model, criterion, optimizer, val_loader, epochs)
        result["model"] = model
        model_name = model._get_name()
        training_history[model_name] = result

    return training_history


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(to_one_hot(y_test))
    )

    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    
    for k, v in mse_configs.items():
        plot_model_guesses(DataLoader(dataset_test), v['model'], k)
        # plt.plot(v['train'], label=f'{k} train')
        # plt.plot(v['val'], label=f'{k} validation')
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.legend(loc='best', prop={'size': 6})
    # plt.show()


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()

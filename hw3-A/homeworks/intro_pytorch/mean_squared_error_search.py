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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


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
    n = 0
    correct = 0

    with torch.no_grad():
        for observation, target in dataloader:
            prediction = model(observation)
            y_hat = torch.argmax(prediction.data, 1)
            y = torch.argmax(target.data, 1)
            correct += (y_hat == y).sum().item()
            n += target.size(0)

    return correct / n


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
    models = {
        'Linear Regression': torch.nn.Sequential(LinearLayer(2, 2, RNG).double()),
        'Hidden -> Sigmoid': torch.nn.Sequential(LinearLayer(2, 2, RNG).double(), SigmoidLayer().double(), LinearLayer(2, 2, RNG).double()),
        'Hidden -> ReLU': torch.nn.Sequential(LinearLayer(2, 2, RNG).double(), ReLULayer().double(), LinearLayer(2, 2, RNG).double()),
        'Hidden -> Sigmoid -> Hidden -> ReLU': torch.nn.Sequential(LinearLayer(2, 2, RNG).double(), SigmoidLayer().double(), LinearLayer(2, 2, RNG).double(), ReLULayer().double(), LinearLayer(2, 2, RNG).double()),
        'Hidden -> ReLU -> Hidden -> Sigmoid': torch.nn.Sequential(LinearLayer(2, 2, RNG).double(), ReLULayer().double(), LinearLayer(2, 2, RNG).double(),SigmoidLayer().double(), LinearLayer(2, 2, RNG).double())
    }
    criterion = MSELossLayer()
    batch_size = 32
    lr = 0.004
    epochs = 100

    for name, model in models.items():
        train_loader = DataLoader(dataset_train, batch_size=batch_size)
        val_loader = DataLoader(dataset_val, batch_size=batch_size)
        optimizer = SGDOptimizer(model.parameters(), lr)

        result = train(train_loader, model, criterion, optimizer, val_loader, epochs)
        result['model'] = model
        training_history[name] = result

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
        plt.plot(v['train'], label=f'{k} train')
        plt.plot(v['val'], label=f'{k} validation')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='best', prop={'size': 6})
    plt.show()

    best_model = min(mse_configs.items(), key=lambda x: min(x[1]['val']))
    print(best_model)
    plot_model_guesses(DataLoader(dataset_test, batch_size=32), best_model[1]['model'])
    
    accuracy = accuracy_score(best_model[1]['model'], DataLoader(dataset_test, batch_size=32))
    print(accuracy)


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
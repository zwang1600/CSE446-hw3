# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()

        alpha_d = 1 / math.sqrt(d)
        alpha_h = 1 / math.sqrt(h)

        distribution_d = Uniform(-alpha_d, alpha_d)
        distribution_h = Uniform(-alpha_h, alpha_h)
        
        self.W_0 = Parameter(distribution_d.sample((h, d)))
        self.W_1 = Parameter(distribution_h.sample((k, h)))

        self.b_0 = Parameter(distribution_d.sample((h,)))
        self.b_1 = Parameter(distribution_h.sample((k,)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        predict = relu(self.W_0 @ x.T + self.b_0.unsqueeze(1))
        predict = relu(self.W_1 @ predict + self.b_1.unsqueeze(1))

        return predict.T


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha_d = 1 / math.sqrt(d)
        alpha_h0 = 1 / math.sqrt(h0)
        alpha_h1 = 1 / math.sqrt(h1)

        distribution_d = Uniform(-alpha_d, alpha_d)
        distribution_h0 = Uniform(-alpha_h0, alpha_h0)
        distribution_h1 = Uniform(-alpha_h1, alpha_h1)
        
        self.W_0 = Parameter(distribution_d.sample((h0, d)))
        self.W_1 = Parameter(distribution_h0.sample((h1, h0)))
        self.W_2 = Parameter(distribution_h1.sample((k, h1)))

        self.b_0 = Parameter(distribution_d.sample((h0,)))
        self.b_1 = Parameter(distribution_h0.sample((h1,)))
        self.b_2 = Parameter(distribution_h1.sample((k,)))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        predict = relu(self.W_0 @ x.T + self.b_0.unsqueeze(1))
        predict = relu(self.W_1 @ predict + self.b_1.unsqueeze(1))
        predict = self.W_2 @ predict + self.b_2.unsqueeze(1)

        return predict.T


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    n = len(train_loader)
    training_loss = []
    accuracy = 0

    while accuracy < 0.99:
        epoch_train_loss = 0
        correct_prediction = 0
        
        for _, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model.forward(x)
            loss = cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if torch.argmax(y_hat) == y: correct_prediction += 1
        
        accuracy = correct_prediction / n
        training_loss.append(epoch_train_loss / n)

    return training_loss


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    d = 784
    h1 = 64
    h2 = 32
    k = 10
    # F2(h2, h2, d, k)
    # F1(h1, d, k)
    train_loader = DataLoader(TensorDataset(x[:500], y[:500]))
    for model in [F2(h2, h2, d, k)]:
        optimizer = Adam(model.parameters(), lr=1e-3)
        training_loss = train(model, optimizer, train_loader)
        plt.plot(training_loss)
        plt.ylabel('Training Loss')
        plt.xlabel('Epochs')
        plt.show()


if __name__ == "__main__":
    main()

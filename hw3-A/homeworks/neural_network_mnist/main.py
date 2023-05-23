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
from tqdm import tqdm

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
        predict = relu(torch.matmul(x, self.W_0.t()) + self.b_0)
        predict = torch.matmul(predict, self.W_1.t()) + self.b_1
        return predict


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
        predict = relu(torch.matmul(x, self.W_0.t()) + self.b_0)
        predict = relu(torch.matmul(predict, self.W_1.t()) + self.b_1)
        predict = torch.matmul(predict, self.W_2.t()) + self.b_2
        return predict


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
        print(accuracy)
        epoch_training_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_training_loss += loss.item()
            correct += (torch.argmax(y_hat, dim=1) == y).sum().item()
            total += y.size(0)

        training_loss.append(epoch_training_loss / n)
        accuracy = correct / total

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

    # F1(h1, d, k)
    # F2(h2, h2, d, k)
    model = F1(h1, d, k)
    train_data = TensorDataset(x, y)
    test_data = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    optimizer = Adam(model.parameters())
    training_loss = train(model, optimizer, train_loader)
    plt.plot(training_loss)
    plt.ylabel('Training Loss')
    plt.xlabel('Epochs')
    plt.title('Loss of F2')
    plt.show()

    # Evaluate on test dataset
    loss = 0
    n = 0
    correct = 0
    
    with torch.no_grad():
        for observation, target in test_loader:
            prediction = model(observation)
            y_hat = torch.argmax(prediction.data, 1)
            correct += (y_hat == target).sum().item()
            n += target.size(0)
            loss += cross_entropy(prediction, target).item()
            
    loss /= len(test_loader)
    accuracy = correct / n

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    num_of_params = sum(param.numel() for param in model.parameters())
    print(f'Number of parameters: {num_of_params}')


if __name__ == "__main__":
    main()
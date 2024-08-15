import torch
import torch.ao.quantization as tq
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os

# Make torch deterministic
_ = torch.manual_seed(0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
padding = nn.ZeroPad2d(18)
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_trainset.data = padding(mnist_trainset.data)
print("outside loader")
print(mnist_trainset.data[0])
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# Load the MNIST test set
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_testset.data = padding(mnist_testset.data)
print(mnist_testset.data.size())

test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

# Define the device
device = "cpu"

# Define the model
class VerySimpleNet(nn.Module):
    def __init__(self):
        super(VerySimpleNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=4,              
                stride=1,                   
                padding=0,
                bias=False                 
            )                                     
        self.ReLu1 = nn.ReLU()    
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.add_padding = nn.ZeroPad2d(17)
        self.conv2 = nn.Conv2d(16, 32, 4, 1, 0, bias=False)  
        self.ReLu2 = nn.ReLU()    
        # fully connected layer, output 10 classes
        self.linear = nn.Linear(32 * 30 * 30, 10)
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x, test):
        if(test):
            print(x.dtype)
            print("prequant")
            plt.imshow(x[0][0], cmap='gray')
            plt.title("prequant")
            plt.show()
            input()
        x = self.quant(x)
        if(test):
            print(x.dtype)
            print("preconv")
            plt.imshow(torch.int_repr(x[0][0]), cmap='gray')
            plt.title("postquant")
            plt.show()
            input()
            torch.save(x, "data/first_conv_relu_in.pt")
        x = self.conv1(x)
        if(test):
            print(x)
            print("prerelu")
            plt.imshow(torch.int_repr(x[0][0]), cmap='gray')
            plt.title("pre-relu")
            plt.show()
            input()
            torch.save(x, "data/first_conv_out.pt")
        x = self.ReLu1(x)
        if(test):
            print(x)
            plt.imshow(torch.int_repr(x[0][0]), cmap='gray')
            plt.title("out")
            plt.show()
            input()
            torch.save(x, "data/first_conv_relu_out.pt")

        x = self.MaxPool(x)
        x = self.add_padding(x)

        x = self.conv2(x)
        x = self.ReLu2(x)
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)       
        x = self.linear(x)
        x = self.dequant(x)
        return  x

net = VerySimpleNet().to(device)

# Insert min-max observers in the model
activation_qconfig = _LearnableFakeQuantize.with_args(
    observer = tq.MinMaxObserver,
    quant_min = 0,
    quant_max = 255,
    dtype = torch.quint8,
    qscheme = torch.per_channel_affine,
    scale = 0.1,
    zero_point = 0.0,
    use_grad_scaling = True
)
net.qconfig = torch.ao.quantization.default_qconfig
net.eval()
net_fused = torch.ao.quantization.fuse_modules(net, [['conv1', 'ReLu1'], ['conv2', 'ReLu2']])
net_quantized = torch.ao.quantization.prepare_qat(net_fused.train()) # Insert observers

def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            torch.set_printoptions(profile="full")
            # print("inside loader")
            # print(torch.round(x[0]))
            # input()
            x = torch.round(x)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.repeat(1, 8, 1, 1), 0)
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return
            
def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0

    iterations = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.repeat(1, 8, 1, 1), 0)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    print(f'Accuracy: {round(correct/total, 3)}')

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

# Train the model
train(train_loader, net_quantized, epochs=1)

# Check the collected statistics during training
print(f'Check statistics of the various layers')
print(net_quantized)

# Quantize the model using the statistics collected
net_quantized.eval()
net_quantized = torch.ao.quantization.convert(net_quantized)

print(f'Check statistics of the various layers')
print(net_quantized)

# Print weights and size of the model after quantization
print('Weights after quantization')
print(torch.int_repr(net_quantized.conv1.weight()))
torch.save(net_quantized.conv1.weight(), "data/wgt_tensor1.pt")
print(torch.int_repr(net_quantized.conv2.weight()))
torch.save(net_quantized.conv2.weight(), "data/wgt_tensor2.pt")

# Test the model after quantization
print('Testing the model after quantization')
test(net_quantized)

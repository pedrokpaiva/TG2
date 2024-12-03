import importlib
import os
import csv
import numpy as np
import torch
import torch.ao.quantization as tq
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
from tqdm import tqdm
from pathlib import Path
from MAxPy import results
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from bitstring import BitArray

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


    def forward(self, x, ckt=None, wgt_tensor=None, results_filename=None, test = 0):
        if(test):
            torch.set_printoptions(profile="full")
            # print(x[0])
            # print("x puro")
            # input()
            # print(x[0].to(torch.int32))
            # print("x int")
            # input()
            x = self.quant(x)  
            # print(x)
            # print("x quant")
            # input()
            # print(torch.int_repr(x))
            # print("x intr_repr")
            # input()
            torch.set_printoptions(profile="default")
        if(test):
            node_info, x = run_rtl(ckt, x, wgt_tensor, results_filename)
            x = torch.unsqueeze(x, 0)
        else:
            x = self.quant(x)  
            x = self.conv1(x)
            x = self.ReLu1(x)

        x = self.MaxPool(x)
        x = self.add_padding(x)
        x = self.conv2(x)
        x = self.ReLu2(x)
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)       
        x = self.linear(x)
        x = self.dequant(x)
        if test:
            return node_info, x
        else:
            return  0, x

def testbench_run(ckt=None, results_filename=None):
    # Make torch deterministic
    _ = torch.manual_seed(0)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset
    padding = nn.ZeroPad2d(18)
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    # print(mnist_trainset.data[0])
    # print("mnist")
    # input()
    mnist_trainset.data = padding(mnist_trainset.data)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=False)
    # for data in train_loader:
    #     x, y = data
    #     print(x[0])
    #     print("trainloader")
    #     input()

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    mnist_testset.data = padding(mnist_testset.data)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=1, shuffle=False)

    # Define the device
    device = "cpu"


    net = VerySimpleNet().to(device)

    # Insert min-max observers in the model
    net.qconfig = torch.ao.quantization.default_qconfig
    net.eval()
    net_fused = torch.ao.quantization.fuse_modules(net, [['conv1', 'ReLu1'], ['conv2', 'ReLu2']])
    net_quantized = torch.ao.quantization.prepare_qat(net_fused.train()) # Insert observers

    # Train the model
    if ( os.path.isfile( "data/net_quantized.pt")):
        train(train_loader, net_quantized, epochs=1)
        # Check the collected statistics during training
        print(f'Check statistics of the various layers')
        print(net_quantized)

        # Quantize the model using the statistics collected
        net_quantized.eval()
        net_quantized = torch.ao.quantization.convert(net_quantized)

        print(f'Check statistics of the various layers')
        print(net_quantized)
        # input()

        # Print weights and size of the model after quantization
        print('Weights after quantization')
        print(torch.int_repr(net_quantized.conv1.weight()))
        torch.save(net_quantized.conv1.weight(), "data/wgt_tensor1.pt")
        print(torch.int_repr(net_quantized.conv2.weight()))
        torch.save(net_quantized.conv2.weight(), "data/wgt_tensor2.pt")

        # Save model
        torch.save(net_quantized.state_dict(), "data/net_quantized.pt")
    else:
        net_quantized = torch.ao.quantization.convert(net_quantized)
        net_quantized.load_state_dict(torch.load("data/net_quantized.pt"))
        net_quantized.eval()

    # Test the model after quantization
    print('Testing the model after quantization')

    #Setup input files and input values
    # ifm_arr_msb =[]
    # ifm_arr_lsb =[]
    # ifm_arr_temp =[]
    # with open("data/ifm.txt") as f:
    #     for line in f:
    #         if not line.isspace():
    #             ifm_arr_temp.append(BitArray(bin="".join(line.split())))
    #             curr_ifm_arr = ifm_arr_temp.pop(0)
    #             ifm_arr_msb.append(curr_ifm_arr[0:32])
    #             ifm_arr_lsb.append(curr_ifm_arr[32:64])

    wgt_tensor = torch.load("data/wgt_tensor1.pt")

    prun_flag, node_info = test(test_loader, net_quantized, ckt=ckt, wgt_tensor=wgt_tensor, results_filename=results_filename)
    return prun_flag, node_info

def run_rtl(ckt=None, ifm_tensor=None,  wgt_tensor=None, results_filename=None):
    # Parameters definition
    CI = 0
    CO = 1
    TI = 16
    TI_FACTOR = 64//TI
    CFG_CI = (CI+1)*8
    CFG_CO = (CO+1)*8
    IFM_LEN = CFG_CI*(TI+3)*TI_FACTOR*13*8
    WGT_LEN = 4*4*CFG_CI*CFG_CO*13*TI_FACTOR
    BUF_DEPTH = 61
    OFM_C = CFG_CO
    OFM_H = BUF_DEPTH
    OFM_W = BUF_DEPTH

    conv = ckt.conv_acc()

    print(f">>> testbench init - circuit: {conv.name()}, area: {conv.area}, parameters: {conv.parameters}")

    # Convert tensor to list to drive RTL
    wgt_transpose = torch.transpose(wgt_tensor, 2, 3)
    wgt_np = torch.int_repr(wgt_transpose).numpy()

    ifm_np = torch.int_repr(ifm_tensor).numpy()
    # print(ifm_np)
    # print("ifm_np")
    # input()


    wgt_arr = []
    for oc in wgt_np:
        for ii in range(13):
            for jj in range(TI_FACTOR):
                for ic in oc:
                    for row in ic:
                        wgt_in = BitArray()
                        for wgt in row:
                            wgt_in.append(BitArray(int8=wgt))
                        wgt_arr.append(wgt_in)

    ifm_arr_msb = []
    ifm_arr_lsb = []
    for ii in range(13):
        for jj in range(TI_FACTOR):
            for oc in ifm_np:
                for ic in oc:
                    for j in range(TI + 3):
                        col = jj*TI + j
                        ifm_in_msb = BitArray()
                        ifm_in_lsb = BitArray()
                        for i in range(8):
                            row = ii*5+i
                            if ((row < 64) and (col < 64)):
                                data = ic[row][col]
                            else:
                                data = 0
                            if(i < 4):
                                ifm_in_msb.append(BitArray(uint8=data))
                            else:
                                ifm_in_lsb.append(BitArray(uint8=data))
                        ifm_arr_msb.append(ifm_in_msb)
                        ifm_arr_lsb.append(ifm_in_lsb)

    
    # Initial Reset
    conv.set_clk(0)
    conv.set_rst_n(1)
    conv.set_start_conv(0)
    conv.set_cfg_ci(CI)
    conv.set_cfg_co(CO)
    conv.set_ifm_msb(0)
    conv.set_ifm_lsb(0)
    conv.set_weight(0)

    clk_switch(conv) # posedge
    clk_switch(conv) # negedge
    clk_switch(conv) # posedge
    conv.set_rst_n(0)
    clk_switch(conv) # negedge
    clk_switch(conv) # posedge
    conv.set_rst_n(1)
    clk_switch(conv) # negedge
    clk_switch(conv) # posedge
    conv.set_start_conv(1)
    clk_switch(conv) # negedge
    clk_switch(conv) # posedge
    conv.set_start_conv(0)
    clk_switch(conv) # negedge
    clk_switch(conv) # posedge

    # Always loop
    ofm = [[[0 for _ in range(OFM_W+3)] for _ in range(OFM_H+5)] for _ in range(OFM_C)]
    tensor_array = np.array([[[0 for _ in range(OFM_W)] for _ in range(OFM_H)] for _ in range(OFM_C)], dtype='i4')
    ifm_count = 0
    wgt_count = 0
    stop_flag = 0
    ow = 0
    oc = 0
    oh = 0
    tw = 0
    thcnt = 0
    stop = 0
    while (not stop):
        # ifm input control
        if (conv.get_ifm_read()):
            conv.set_ifm_msb(ifm_arr_msb[ifm_count].int)
            conv.set_ifm_lsb(ifm_arr_lsb[ifm_count].int)
            ifm_count += 1
        else:
            conv.set_ifm_msb(0)
            conv.set_ifm_lsb(0)
            if (ifm_count == IFM_LEN/8):
                ifm_count  = 0

        # weight input control
        if (conv.get_wgt_read()):
            conv.set_weight(wgt_arr[wgt_count].int)
            wgt_count += 1
        else:
            conv.set_weight(0)
            if (wgt_count == WGT_LEN/4):
                wgt_count = 0

        # ofm output control
        if (oc <= (OFM_C - 1)):
            if (conv.get_ofm_port0_v() and conv.get_ofm_port1_v()):
                ofm[oc][oh][ow+tw*TI] = conv.get_ofm_port0()
                ofm[oc][oh+1][ow+tw*TI] = conv.get_ofm_port1()

                ow = ow + 1
                if (ow == TI):
                    ow = 0
                    oh += 2
                    thcnt += 2
            elif (conv.get_ofm_port0_v()):
                ofm[oc][oh][ow+tw*TI] = conv.get_ofm_port0()

                ow = ow + 1
                if (ow == TI):
                    ow = 0
                    oh += 1
                    thcnt += 1
                    if (thcnt == 5):
                        thcnt = 0
                        tw += 1
                        oh -= 5
                        if (tw == 4):
                            tw = 0
                            oh += 5
                if (oh == 65):
                    oh = 0
                    oc += 1
                    print ("Computing channel ", oc)
            stop_flag = 1
        else:
            if (stop_flag):
                with open("data/output_rtl.txt", "w") as f:
                    for toc in range(0, OFM_C):
                        for toh in range(0, OFM_H):
                            buffer = 4 - 2*(toh%5)         # Used to correct rows that come inverted 5 by 5
                            for tow in range(0, OFM_W):
                                s = str(ofm[toc][toh+buffer][tow]) + ","
                                f.write(s)
                                bit_array = BitArray(int=ofm[toc][toh+buffer][tow], length=32)
                                tensor_array[toc][toh][tow] = bit_array.int32

                                if (tow == OFM_W-1):
                                    f.write("\n")
                        f.write("\n")
                print("Finish writing results")
            stop = 1

        clk_switch(conv) # negedge
        clk_switch(conv) # posedge
        # print ("clk ", conv.get_clk())

    with open("data/output_rtl_tensor.txt", "w") as f:
        for i in range(CFG_CO):
            for j in range(BUF_DEPTH):
                for k in tensor_array[i, j, :]:
                    s = str(k) + ","
                    f.write(s)
                f.write("\n")
            f.write("\n")

    # Print tensor_array
    tensor = torch.from_numpy(tensor_array).to(torch.float32)
    # quantized_tensor = torch.quantize_per_tensor(tensor, 0.10757868736982346, 0, torch.quint8) # no round in training transforms.Normalize((0.1307,), (0.3081,))]) acc 0.75
    # quantized_tensor = torch.quantize_per_tensor(tensor, 0.1134602427482605, 0, torch.quint8) # round in training transforms.Normalize((0.1307,), (0.3081,))]) acc 0,861
    # quantized_tensor = torch.quantize_per_tensor(tensor, 0.0721592977643013, 0, torch.quint8) # round in training transforms.Normalize((0.5,), (0.5,))]) acc 0,366
    # quantized_tensor = torch.quantize_per_tensor(tensor, 0.0741245225071907, 0, torch.quint8) # no round in training transforms.Normalize((0.5,), (0.5,))]) acc 0.416
    # quantized_tensor = torch.quantize_per_tensor(tensor, 0.049549445509910583, 0, torch.quint8) # no round in training transforms.ToTensor() acc 0.792
    quantized_tensor = torch.quantize_per_tensor(tensor, 0.04704698547720909, 0, torch.quint8) # round in training transforms.ToTensor() acc 0.881
    # print(quantized_tensor.dtype)
    # print(quantized_tensor)
    # input()


    # # Start comparison
    # golden_result   = csv.reader(open("data/ofm.txt"), delimiter=',')
    # rtl_result      = csv.reader(open("data/output_rtl_tensor.txt"), delimiter=',')

    # golden_vec = []
    # for line in golden_result:
    #     if (line == []):
    #         continue
    #     for item in line:
    #         if item != "":
    #             golden_vec.append(int(item))

    # rtl_vec = []
    # for line in rtl_result:
    #     if (line == []):
    #         continue
    #     for item in line:
    #         if item != "":
    #             rtl_vec.append(int(item))


    # print ("Testing", len(rtl_vec), "[RTL results] ",  len(golden_vec), "[Golden results] ", "values")

    # mae       = mean_absolute_error(golden_vec, rtl_vec)
    # accuracy  = accuracy_score(golden_vec, rtl_vec)

    # for i in range(len(golden_vec)):
    #     if (golden_vec[i] == 0):
    #         golden_vec[i] = 1
    #     if (rtl_vec[i] == 0):
    #         rtl_vec[i] = 1
    # mape = mean_absolute_percentage_error(golden_vec, rtl_vec)

    # rst.add(conv, {"mae": mae, "mape": mape, "accuracy": accuracy})
    # print(f"> mae: {mae:.4f}, mape: {mape:.4f}, accuracy: {accuracy:.4f}")
    # print(">>> Testbench end")

    # result = 0
    # for i in range(len(rtl_vec)):
    #     if (len(rtl_vec[i]) != len(golden_vec[i])):
    #         print (i, " ", len(rtl_vec[i]), " ",   len(golden_vec[i]))
    #     for j in range(len(rtl_vec[i])):
    #         if (rtl_vec[i][j] != golden_vec[i][j]):
    #             result = 1
    #             print ("Opps! wrong result", rtl_vec[i][j], ", ", golden_vec[i][j])

    # if (not result):
    #     print ("Pass, Correct result !!!")
    # else:
    #     print ("Opss, Wrong result !!!")

    # Return prun_flag and node_info
    
    return conv.node_info, quantized_tensor

def train(train_loader, net, epochs=5, total_iterations_limit=None, device='cpu'):
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
            x = torch.round(x)
            y = y.to(device)
            optimizer.zero_grad()
            a, output = net(x.repeat(1, 8, 1, 1), 0)
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

def test(test_loader, model: nn.Module, total_iterations: int = None, device='cpu', ckt=None, ifm_arr_msb=None, ifm_arr_lsb=None,  wgt_tensor=None, results_filename=None):
    correct = 0
    total = 0

    iterations = 0

    model.eval()
    conv = ckt.conv_acc()
    rst = results.ResultsTable(results_filename, ["accuracy"])

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            # x = torch.round(x)
            y = y.to(device)
            # print(x[0])
            # print("x puro")
            # input()
            node_info, output = model(x.repeat(1, 8, 1, 1), ckt, wgt_tensor, results_filename, 1)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y:
                    correct +=1
                total +=1
            iterations += 1
            # if total_iterations is not None and iterations >= total_iterations:
            if(iterations > 100):
                break
    accuracy_score = round(correct/total, 3)
    rst.add(conv, {"accuracy": accuracy_score})
    print(f'Accuracy: {accuracy_score}')

    if (accuracy_score > 0.7):
        prun_flag = True
    else:
        prun_flag = False
    return prun_flag, node_info

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp_delme.p")
    print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
    os.remove('temp_delme.p')

def clk_switch(conv):
    conv.set_clk(0) if conv.get_clk() else conv.set_clk(1)
    conv.eval()

if __name__ == "__main__":
    mod = importlib.import_module(name="conv_acc_exact.CONV_ACC")
    testbench_run(ckt=mod, results_filename="testbench_dev.csv")

import importlib
from MAxPy import results
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from bitstring import BitArray
import csv
import torch
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #RTL
        self.conv1 = nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        self.ReLu1 = nn.ReLU()
        # Truncar pra 8bits

        # SOftware
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)
        #30 ou 31 bits

        #Adicionar padding 64x64
        # self.add_padding = nn.ZeroPad2d()

        #RTL
        self.conv2 = nn.Conv2d(16, 32, 4, 1, 0, bias=False)
        self.ReLu2 = nn.ReLU()

        # Truncar pra 8bits
        # SOftware
        self.MaxPool2 = nn.MaxPool2d(2)
        #30 ou 31 bits

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 30 * 30, 10)

    def forward(self, ckt=None, ifm_arr_msb=None, ifm_arr_lsb=None,  wgt_arr=None, results_filename=None):
        a, b, x = run_rtl(ckt, ifm_arr_msb, ifm_arr_lsb, wgt_arr, results_filename)
        x = self.MaxPool1(x)

        x = self.conv2(x)
        x = self.ReLu2(x)
        x = self.MaxPool2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        # output = self.out(x)
        return a, b, x
        # return output, x    # return x for visualization

def testbench_run(ckt=None, results_filename=None):
    # ifm = torch.rand(1, ic, ih, iw)*255-128
    # ifm = torch.round(ifm)

    # Create CNN
    cnn = CNN()
    cnn.conv2.weight = torch.load("data/wgt_tensor2.pt")


    #Setup input files and input values
    ifm_arr_msb =[]
    ifm_arr_lsb =[]
    ifm_arr_temp =[]
    wgt_arr =[]
    with open("data/ifm.txt") as f:
        for line in f:
            if not line.isspace():
                ifm_arr_temp.append(BitArray(bin="".join(line.split())))
                curr_ifm_arr = ifm_arr_temp.pop(0)
                ifm_arr_msb.append(curr_ifm_arr[0:32])
                ifm_arr_lsb.append(curr_ifm_arr[32:64])

    with open("data/weight.txt") as f:
        for line in f:
            if not line.isspace():
                wgt_arr.append(BitArray(bin="".join(line.split())))

    wgt_tensor = torch.load("data/wgt_tensor1.pt")

    prun_flag, node_info, tensor_out = cnn(ckt, ifm_arr_msb, ifm_arr_lsb, wgt_tensor, results_filename)
    print(tensor_out)
    print(tensor_out.dtype)
    accuracy = Acc
    return prun_flag, node_info



def clk_switch(conv):
    conv.set_clk(0) if conv.get_clk() else conv.set_clk(1)
    conv.eval()

def run_rtl(ckt=None, ifm_arr_msb=None, ifm_arr_lsb=None,  wgt_tensor=None, results_filename=None):
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
    rst = results.ResultsTable(results_filename, ["mae", "mape", "accuracy"])

    print(f">>> testbench init - circuit: {conv.name()}, area: {conv.area}, parameters: {conv.parameters}")

    # Convert tensor to list to drive RTL
    print(wgt_tensor.size())
    wgt_transpose = torch.transpose(wgt_tensor, 2, 3)
    wgt_np = torch.int_repr(wgt_transpose.data).numpy()
    print(wgt_np)


    wgt_arr = []
    # print("weight np")
    # print(len(wgt_np))
    for oc in wgt_np:
        # print(len(oc))
        for ii in range(13):
            for jj in range(TI_FACTOR):
                for ic in oc:
                    # print("ic")
                    # print(len(ic))
                    for row in ic:
                        # print("row")
                        # print(len(row))
                        # input()
                        wgt_in = BitArray()
                        for wgt in row:
                            wgt_in.append(BitArray(int8=wgt))
                        wgt_arr.append(wgt_in)

    # ifm_np = ifm_tensor.data.numpy().astype(int)
    # for ii in range(13):
    #     for jj in range(TI_FACTOR):
    #         for c in range(CFG_CI):
    #             for j in range(TI + 3):
    #                 col = jj*TI + j
    #                 for i in range(8):
    #                     row = ii*5+i
    #                     # print(row, c, ii)
    #                     k = ifm_np[0, c, row, col] if ((row < 64) and (col < 64))else 0
    #                     s = np.binary_repr(k, 8) + " "


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
    tensor_array = np.array([[[0 for _ in range(OFM_W)] for _ in range(OFM_H)] for _ in range(OFM_C)], dtype='u1')
    ifm_count = 0
    wgt_count = 0
    stop_flag = 0
    ow = 0
    oc = 0
    oh = 0
    tw = 0
    thcnt = 0
    stop = 0
    print(len(wgt_arr))
    input()
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
                                bit_array = BitArray(int=ofm[toc][toh+buffer][tow], length=25)
                                del bit_array[0:17]     # truncates output to 8 bit data
                                tensor_array[toc][toh][tow] = bit_array.int8

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
    tensor = torch.from_numpy(tensor_array)
    print(tensor.dtype)
    print(tensor)
    input()
    print(torch.int_repr(torch.load("data/first_conv_relu_out.pt")))
    input()



    # Start comparison
    golden_result   = csv.reader(open("data/ofm.txt"), delimiter=',')
    rtl_result      = csv.reader(open("data/output_rtl_tensor.txt"), delimiter=',')

    golden_vec = []
    for line in golden_result:
        if (line == []):
            continue
        for item in line:
            if item != "":
                golden_vec.append(int(item))

    rtl_vec = []
    for line in rtl_result:
        if (line == []):
            continue
        for item in line:
            if item != "":
                rtl_vec.append(int(item))


    print ("Testing", len(rtl_vec), "[RTL results] ",  len(golden_vec), "[Golden results] ", "values")

    mae       = mean_absolute_error(golden_vec, rtl_vec)
    accuracy  = accuracy_score(golden_vec, rtl_vec)

    for i in range(len(golden_vec)):
        if (golden_vec[i] == 0):
            golden_vec[i] = 1
        if (rtl_vec[i] == 0):
            rtl_vec[i] = 1
    mape = mean_absolute_percentage_error(golden_vec, rtl_vec)

    rst.add(conv, {"mae": mae, "mape": mape, "accuracy": accuracy})
    print(f"> mae: {mae:.4f}, mape: {mape:.4f}, accuracy: {accuracy:.4f}")
    print(">>> Testbench end")

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
    if (mape < 0.1):
        prun_flag = True
    else:
        prun_flag = False

    return prun_flag, conv.node_info, tensor


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


if __name__ == "__main__":
    mod = importlib.import_module(name="conv_acc_exact.CONV_ACC")
    testbench_run(ckt=mod, results_filename="testbench_dev.csv")


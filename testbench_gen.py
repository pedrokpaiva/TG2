import torch
import torch.nn as nn
import numpy as np


ic = 8
ih = 64
iw = 64

oc = 16
oh = 61
ow = 61

kk = 4

conv2d = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=kk, padding=0, bias=False)
relu = nn.ReLU(inplace=False)

# randomize input feature map
ifm = torch.rand(1, ic, ih, iw)*255-128
#ifm = torch.ones(1, ic, ih, iw)
ifm = torch.round(ifm)
# randomize weight
weight = torch.rand(oc, ic, kk, kk)*255 - 128
# weight = torch.rand(oc, ic, kk, kk)*4
# weight = torch.ones(oc, ic, kk, kk)
# weight = torch.randint(1,4,(oc, ic, kk, kk))
weight = torch.round(weight)

# setting the kernel of conv2d as weight
conv2d.weight = nn.Parameter(weight)

# computing output feature
ofm = conv2d(ifm)
ofm_relu = relu(ofm)

ifm = torch.load("first_conv_relu_in.pt")
weight = torch.load("wgt_tensor.pt")
ofm_relu = torch.load("first_conv_relu_out.pt")

torch.set_printoptions(profile="default")
print(ifm)
print(torch.int_repr(ifm))
input()


ifm_np =  torch.int_repr(ifm.data).numpy()
weight_np = torch.int_repr(weight.data).numpy()
ofm_np = torch.int_repr(ofm_relu.data).numpy()

# write data as a 2's complement binary representation type
with open("ifm_bin_c%dxh%dxw%d.txt"%(ic, ih, iw), "w") as f:
    for i in range(ic):
        for j in range(ih):
            for k in ifm_np[0, i, j, :]:
                s = np.binary_repr(k, 8) + " "
                f.write(s)
            f.write("\n")
        f.write("\n")


with open("ofm_bin_c%dxh%dxw%d.txt"%(oc, oh, ow), "w") as f:
    for i in range(oc):
        for j in range(oh):
            for k in ofm_np[0, i, j, :]:
                s = np.binary_repr(k, 25) + " "
                f.write(s)
            f.write("\n")
        f.write("\n")


with open("weight_bin_co%dxci%dxk%dxk%d.txt"%(oc, ic, kk, kk), "w") as f:
    for i in range(oc):
        for j in range(ic):
            for k in range(kk):
                for l in weight_np[i, j, k, :]:
                    s = np.binary_repr(l, 8) + " "
                    f.write(s)
                f.write("\n")
            f.write("\n")
        f.write("\n")


# write out data as decimal type
with open("ifm_dec_%dxh%dxw%d.txt" % (ic, ih, iw), "w") as f:
    for i in range(ic):
        for j in range(ih):
            for k in ifm_np[0, i, j, :]:
                s = str(k) + "\t "
                f.write(s)
            f.write("\n")
        f.write("\n")

with open("ofm.txt", "w") as f:
    for i in range(oc):
        for j in range(oh):
            for k in ofm_np[0, i, j, :]:
                s = str(k) + ","
                f.write(s)
            f.write("\n")
        f.write("\n")

with open("weight_dec_co%dxci%dxk%dxk%d.txt" % (oc, ic, kk, kk), "w") as f:
    for i in range(oc):
        for j in range(ic):
            for k in range(kk):
                for l in weight_np[i, j, k, :]:
                    s = str(l) + " "
                    f.write(s)
                f.write("\n")
            f.write("\n")
        f.write("\n")

tile_length = 16
num_tile = 64//tile_length

with open("ifm.txt", "w") as f:
    for ii in range(13):
        for jj in range(num_tile):
            for c in range(ic):
                for j in range(tile_length + 3):
                    col = jj*tile_length + j
                    for i in range(8):
                        row = ii*5+i
                        # print(row, c, ii)
                        k = ifm_np[0, c, row, col] if ((row < 64) and (col < 64))else 0
                        s = np.binary_repr(k, 8) + " "
                        f.write(s)
                    f.write("\n")
            f.write("\n")    
        f.write("\n")
    f.write("\n")
    

with open("weight.txt", "w") as f:
    for i in range(oc):
        for ii in range(13):
            for jj in range(num_tile):
                for j in range(ic):
                    for k in range(kk):
                        for l in weight_np[i, j, :, k]:
                            s = np.binary_repr(l, 8) + " "
                            f.write(s)
                        f.write("\n")
                    f.write("\n")
                f.write("\n")
            f.write("\n")
        f.write("\n")

with open("ifm_d_c%dxh%dxw%d.txt"%(ic, ih, iw), "w") as f:
    for ii in range(13):
        for jj in range(num_tile):
            for c in range(ic):
                for j in range(tile_length + 3):
                    col = jj*tile_length + j
                    for i in range(8):
                        row = ii*5+i
                        # print(row, c, ii)
                        k = ifm_np[0, c, row, col] if ((row < 64) and (col < 64)) else 0
                        s = str(k) + " "
                        f.write(s)
                    f.write("\n")
            f.write("\n")    
        f.write("\n")
    f.write("\n")
    

with open("weight_d_co%dxci%dxk%dxk%d.txt"%(oc, ic, kk, kk), "w") as f:
    for i in range(oc):
        for ii in range(13):
            for jj in range(num_tile):
                for j in range(ic):
                    for k in range(kk):
                        for l in weight_np[i, j, :, k]:
                            s = str(l) + " "
                            f.write(s)
                        f.write("\n")
                    f.write("\n")
                f.write("\n")
            f.write("\n")
        f.write("\n")

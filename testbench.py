import importlib
from MAxPy import results
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from bitstring import BitArray
import csv


def testbench_run(ckt=None, results_filename=None):

    # Parameters definition
    CI = 0
    CO = 0
    TI = 16
    TI_FACTOR = 64/TI
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

    #Setup input files and input values
    ifm_arr_msb = []
    ifm_arr_lsb = []
    ifm_arr_temp = []
    wgt_arr = []
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
                    # print ("Computing channel ", oc)
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
                                if (tow == OFM_W-1):
                                    f.write("\n")
                        f.write("\n")
                # print("Finish writing results")
            stop = 1

        clk_switch(conv) # negedge
        clk_switch(conv) # posedge
        # print ("clk ", conv.get_clk())

    # Start comparison
    golden_result   = csv.reader(open("data/ofm.txt"), delimiter=',')
    rtl_result      = csv.reader(open("data/output_rtl.txt"), delimiter=',')

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


    # print ("Testing", len(rtl_vec), "[RTL results] ",  len(golden_vec), "[Golden results] ", "values")

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

    return prun_flag, conv.node_info

def clk_switch(conv):
    conv.set_clk(0) if conv.get_clk() else conv.set_clk(1)
    conv.eval()


if __name__ == "__main__":
    mod = importlib.import_module(name="conv_acc_exact.CONV_ACC")
    testbench_run(ckt=mod, results_filename="testbench_dev.csv")


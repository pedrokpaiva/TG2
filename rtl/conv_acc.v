///==------------------------------------------------------------------==///
/// Conv kernel: top level module
///==------------------------------------------------------------------==///

module conv_acc #(
    parameter out_data_width = 25,
    parameter buf_addr_width = 5,
    parameter buf_depth      = 16
) (
    input  clk,
    input  rst_n,
    input  start_conv,
    input  [1:0] cfg_ci,
    input  [1:0] cfg_co,
    input  [31:0] ifm_msb,
    input  [31:0] ifm_lsb,
    input  [31:0] weight,
    output [24:0] ofm_port0,
    output [24:0] ofm_port1,
    output ofm_port0_v,
    output ofm_port1_v,
    output ifm_read,
    output wgt_read,
    output end_conv
);
    wire [63:0] ifm;

    assign ifm = {ifm_msb, ifm_lsb};
    /// Assign ifm to each pes
    reg [7:0] rows [0:7];
    always @(*) begin
        rows[0] = ifm[7:0];
        rows[1] = ifm[15:8];
        rows[2] = ifm[23:16];
        rows[3] = ifm[31:24];
        rows[4] = ifm[39:32];
        rows[5] = ifm[47:40];
        rows[6] = ifm[55:48];
        rows[7] = ifm[63:56];
        // {rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7]} = 
        //     {ifm[7:0], ifm[15:8], ifm[23:16], ifm[31:24], ifm[39:32], ifm[47:40], ifm[55:48], ifm[63:56]};
    end
    /// Assign weight to each pes
    reg [7:0] wgts [0:3];
    always @(*) begin
        wgts[0] = weight[7:0];
        wgts[1] = weight[15:8];
        wgts[2] = weight[23:16];
        wgts[3] = weight[31:24];

        // {wgts[0], wgts[1], wgts[2], wgts[3]} = {weight[7:0], weight[15:8], weight[23:16], weight[31:24]};
    end

    ///==-------------------------------------------------------------------------------------==
    /// Connect between PE and PE_FSM
    wire ifm_read_en;
    wire wgt_read_en;
    assign ifm_read = ifm_read_en;
    assign wgt_read = wgt_read_en;
    /// Connection between PEs+PE_FSM and WRITEBACK+BUFF
    wire [out_data_width-1:0] pe00_data, pe10_data, pe20_data, pe30_data;
    wire [out_data_width-1:0] pe01_data, pe11_data, pe21_data, pe31_data;
    wire [out_data_width-1:0] pe02_data, pe12_data, pe22_data, pe32_data;
    wire [out_data_width-1:0] pe03_data, pe13_data, pe23_data, pe33_data;
    wire [out_data_width-1:0] pe04_data, pe14_data, pe24_data, pe34_data;
    wire p_filter_end, p_valid_data, start_again;
    /// PE FSM
    PE_FSM pe_fsm ( .clk(clk), .rst_n(rst_n), .start_conv(start_conv), .start_again(start_again), .cfg_ci(cfg_ci), .cfg_co(cfg_co), 
            .ifm_read(ifm_read_en), .wgt_read(wgt_read_en), .p_valid_output(p_valid_data), 
            .last_chanel_output(p_filter_end), .end_conv(end_conv) );  
    
    /// PE Array
    /// wgt0 row0 pe00 pe01 pe02 pe03 pe04
    /// wgt1 row1 pe10 pe11 pe12 pe13 pe14
    /// wgt2 row2 pe20 pe21 pe22 pe23 pe24
    /// wgt3      pe30 pe31 pe32 pe33 pe34
    ///      row3      row4 row5 row6 row7

    /// First row
    wire [7:0] ifm_buf00, ifm_buf01, ifm_buf02, ifm_buf03;
    wire [7:0] ifm_buf10, ifm_buf11, ifm_buf12, ifm_buf13;
    wire [7:0] ifm_buf20, ifm_buf21, ifm_buf22, ifm_buf23;
    wire [7:0] ifm_buf30, ifm_buf31, ifm_buf32, ifm_buf33;
    wire [7:0] ifm_buf40, ifm_buf41, ifm_buf42, ifm_buf43;
    wire [7:0] ifm_buf50, ifm_buf51, ifm_buf52, ifm_buf53;
    wire [7:0] ifm_buf60, ifm_buf61, ifm_buf62, ifm_buf63;
    wire [7:0] ifm_buf70, ifm_buf71, ifm_buf72, ifm_buf73;

	wire [7:0] wgt_buf00, wgt_buf01, wgt_buf02, wgt_buf03;
	wire [7:0] wgt_buf10, wgt_buf11, wgt_buf12, wgt_buf13;
	wire [7:0] wgt_buf20, wgt_buf21, wgt_buf22, wgt_buf23;
	wire [7:0] wgt_buf30, wgt_buf31, wgt_buf32, wgt_buf33;



	IFM_BUF ifm_buf0_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[0]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf00), .ifm_buf1(ifm_buf01), .ifm_buf2(ifm_buf02), .ifm_buf3(ifm_buf03));
	IFM_BUF ifm_buf1_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[1]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf10), .ifm_buf1(ifm_buf11), .ifm_buf2(ifm_buf12), .ifm_buf3(ifm_buf13));
	IFM_BUF ifm_buf2_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[2]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf20), .ifm_buf1(ifm_buf21), .ifm_buf2(ifm_buf22), .ifm_buf3(ifm_buf23));
	IFM_BUF ifm_buf3_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[3]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf30), .ifm_buf1(ifm_buf31), .ifm_buf2(ifm_buf32), .ifm_buf3(ifm_buf33));
	IFM_BUF ifm_buf4_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[4]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf40), .ifm_buf1(ifm_buf41), .ifm_buf2(ifm_buf42), .ifm_buf3(ifm_buf43));
	IFM_BUF ifm_buf5_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[5]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf50), .ifm_buf1(ifm_buf51), .ifm_buf2(ifm_buf52), .ifm_buf3(ifm_buf53));
	IFM_BUF ifm_buf6_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[6]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf60), .ifm_buf1(ifm_buf61), .ifm_buf2(ifm_buf62), .ifm_buf3(ifm_buf63));
	IFM_BUF ifm_buf7_m( .clk(clk), .rst_n(rst_n), .ifm_input(rows[7]), .ifm_read(ifm_read_en), 
	.ifm_buf0(ifm_buf70), .ifm_buf1(ifm_buf71), .ifm_buf2(ifm_buf72), .ifm_buf3(ifm_buf73));

	WGT_BUF wgt_buf0_m( .clk(clk), .rst_n(rst_n), .wgt_input(wgts[0]), .wgt_read(wgt_read_en), 
	.wgt_buf0(wgt_buf00), .wgt_buf1(wgt_buf01), .wgt_buf2(wgt_buf02), .wgt_buf3(wgt_buf03));
	WGT_BUF wgt_buf1_m( .clk(clk), .rst_n(rst_n), .wgt_input(wgts[1]), .wgt_read(wgt_read_en), 
	.wgt_buf0(wgt_buf10), .wgt_buf1(wgt_buf11), .wgt_buf2(wgt_buf12), .wgt_buf3(wgt_buf13));
	WGT_BUF wgt_buf2_m( .clk(clk), .rst_n(rst_n), .wgt_input(wgts[2]), .wgt_read(wgt_read_en), 
	.wgt_buf0(wgt_buf20), .wgt_buf1(wgt_buf21), .wgt_buf2(wgt_buf22), .wgt_buf3(wgt_buf23));
	WGT_BUF wgt_buf3_m( .clk(clk), .rst_n(rst_n), .wgt_input(wgts[3]), .wgt_read(wgt_read_en), 
	.wgt_buf0(wgt_buf30), .wgt_buf1(wgt_buf31), .wgt_buf2(wgt_buf32), .wgt_buf3(wgt_buf33));
    
	PE pe00( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf00), .ifm_input1(ifm_buf01), .ifm_input2(ifm_buf02), .ifm_input3(ifm_buf03), 
	.wgt_input0(wgt_buf00), .wgt_input1(wgt_buf01), .wgt_input2(wgt_buf02), .wgt_input3(wgt_buf03), .p_sum(pe00_data) );
	PE pe01( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf10), .ifm_input1(ifm_buf11), .ifm_input2(ifm_buf12), .ifm_input3(ifm_buf13), 
	.wgt_input0(wgt_buf00), .wgt_input1(wgt_buf01), .wgt_input2(wgt_buf02), .wgt_input3(wgt_buf03), .p_sum(pe01_data) );
	PE pe02( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf20), .ifm_input1(ifm_buf21), .ifm_input2(ifm_buf22), .ifm_input3(ifm_buf23), 
	.wgt_input0(wgt_buf00), .wgt_input1(wgt_buf01), .wgt_input2(wgt_buf02), .wgt_input3(wgt_buf03), .p_sum(pe02_data) );
	PE pe03( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf30), .ifm_input1(ifm_buf31), .ifm_input2(ifm_buf32), .ifm_input3(ifm_buf33), 
	.wgt_input0(wgt_buf00), .wgt_input1(wgt_buf01), .wgt_input2(wgt_buf02), .wgt_input3(wgt_buf03), .p_sum(pe03_data) );
	PE pe04( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf40), .ifm_input1(ifm_buf41), .ifm_input2(ifm_buf42), .ifm_input3(ifm_buf43), 
	.wgt_input0(wgt_buf00), .wgt_input1(wgt_buf01), .wgt_input2(wgt_buf02), .wgt_input3(wgt_buf03), .p_sum(pe04_data) );


	PE pe10( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf10), .ifm_input1(ifm_buf11), .ifm_input2(ifm_buf12), .ifm_input3(ifm_buf13), 
	.wgt_input0(wgt_buf10), .wgt_input1(wgt_buf11), .wgt_input2(wgt_buf12), .wgt_input3(wgt_buf13), .p_sum(pe10_data) );
	PE pe11( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf20), .ifm_input1(ifm_buf21), .ifm_input2(ifm_buf22), .ifm_input3(ifm_buf23), 
	.wgt_input0(wgt_buf10), .wgt_input1(wgt_buf11), .wgt_input2(wgt_buf12), .wgt_input3(wgt_buf13), .p_sum(pe11_data) );
	PE pe12( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf30), .ifm_input1(ifm_buf31), .ifm_input2(ifm_buf32), .ifm_input3(ifm_buf33), 
	.wgt_input0(wgt_buf10), .wgt_input1(wgt_buf11), .wgt_input2(wgt_buf12), .wgt_input3(wgt_buf13), .p_sum(pe12_data) );
	PE pe13( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf40), .ifm_input1(ifm_buf41), .ifm_input2(ifm_buf42), .ifm_input3(ifm_buf43), 
	.wgt_input0(wgt_buf10), .wgt_input1(wgt_buf11), .wgt_input2(wgt_buf12), .wgt_input3(wgt_buf13), .p_sum(pe13_data) );
	PE pe14( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf50), .ifm_input1(ifm_buf51), .ifm_input2(ifm_buf52), .ifm_input3(ifm_buf53), 
	.wgt_input0(wgt_buf10), .wgt_input1(wgt_buf11), .wgt_input2(wgt_buf12), .wgt_input3(wgt_buf13), .p_sum(pe14_data) );


	PE pe20( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf20), .ifm_input1(ifm_buf21), .ifm_input2(ifm_buf22), .ifm_input3(ifm_buf23), 
	.wgt_input0(wgt_buf20), .wgt_input1(wgt_buf21), .wgt_input2(wgt_buf22), .wgt_input3(wgt_buf23), .p_sum(pe20_data) );
	PE pe21( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf30), .ifm_input1(ifm_buf31), .ifm_input2(ifm_buf32), .ifm_input3(ifm_buf33), 
	.wgt_input0(wgt_buf20), .wgt_input1(wgt_buf21), .wgt_input2(wgt_buf22), .wgt_input3(wgt_buf23), .p_sum(pe21_data) );
	PE pe22( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf40), .ifm_input1(ifm_buf41), .ifm_input2(ifm_buf42), .ifm_input3(ifm_buf43), 
	.wgt_input0(wgt_buf20), .wgt_input1(wgt_buf21), .wgt_input2(wgt_buf22), .wgt_input3(wgt_buf23), .p_sum(pe22_data) );
	PE pe23( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf50), .ifm_input1(ifm_buf51), .ifm_input2(ifm_buf52), .ifm_input3(ifm_buf53), 
	.wgt_input0(wgt_buf20), .wgt_input1(wgt_buf21), .wgt_input2(wgt_buf22), .wgt_input3(wgt_buf23), .p_sum(pe23_data) );
	PE pe24( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf60), .ifm_input1(ifm_buf61), .ifm_input2(ifm_buf62), .ifm_input3(ifm_buf63), 
	.wgt_input0(wgt_buf20), .wgt_input1(wgt_buf21), .wgt_input2(wgt_buf22), .wgt_input3(wgt_buf23), .p_sum(pe24_data) );


	PE pe30( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf30), .ifm_input1(ifm_buf31), .ifm_input2(ifm_buf32), .ifm_input3(ifm_buf33), 
	.wgt_input0(wgt_buf30), .wgt_input1(wgt_buf31), .wgt_input2(wgt_buf32), .wgt_input3(wgt_buf33), .p_sum(pe30_data) );
	PE pe31( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf40), .ifm_input1(ifm_buf41), .ifm_input2(ifm_buf42), .ifm_input3(ifm_buf43), 
	.wgt_input0(wgt_buf30), .wgt_input1(wgt_buf31), .wgt_input2(wgt_buf32), .wgt_input3(wgt_buf33), .p_sum(pe31_data) );
	PE pe32( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf50), .ifm_input1(ifm_buf51), .ifm_input2(ifm_buf52), .ifm_input3(ifm_buf53), 
	.wgt_input0(wgt_buf30), .wgt_input1(wgt_buf31), .wgt_input2(wgt_buf32), .wgt_input3(wgt_buf33), .p_sum(pe32_data) );
	PE pe33( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf60), .ifm_input1(ifm_buf61), .ifm_input2(ifm_buf62), .ifm_input3(ifm_buf63), 
	.wgt_input0(wgt_buf30), .wgt_input1(wgt_buf31), .wgt_input2(wgt_buf32), .wgt_input3(wgt_buf33), .p_sum(pe33_data) );
	PE pe34( .clk(clk), .rst_n(rst_n), .ifm_input0(ifm_buf70), .ifm_input1(ifm_buf71), .ifm_input2(ifm_buf72), .ifm_input3(ifm_buf73), 
	.wgt_input0(wgt_buf30), .wgt_input1(wgt_buf31), .wgt_input2(wgt_buf32), .wgt_input3(wgt_buf33), .p_sum(pe34_data) );	

    ///==-------------------------------------------------------------------------------------==
    /// Connection between the buffer and write back controllers
    wire [out_data_width-1:0] fifo_out[0:4];
    wire valid_fifo_out[0:4];
    wire p_write_zero[0:4];
    wire p_init;
    wire odd_cnt;

    /// Write back controller
    WRITE_BACK #(
        .data_width(out_data_width),
        .depth(buf_depth)
    ) writeback_control (
        .clk(clk),
        .rst_n(rst_n),
        .start_init(start_conv),
        .p_filter_end(p_filter_end),
        .row0(fifo_out[0]),
        .row0_valid(valid_fifo_out[0]),
        .row1(fifo_out[1]),
        .row1_valid(valid_fifo_out[1]),
        .row2(fifo_out[2]),
        .row2_valid(valid_fifo_out[2]),
        .row3(fifo_out[3]),
        .row3_valid(valid_fifo_out[3]),
        .row4(fifo_out[4]),
        .row4_valid(valid_fifo_out[4]),
        .p_write_zero0(p_write_zero[0]),
        .p_write_zero1(p_write_zero[1]),
        .p_write_zero2(p_write_zero[2]),
        .p_write_zero3(p_write_zero[3]),
        .p_write_zero4(p_write_zero[4]),
        .p_init(p_init),
        .out_port0(ofm_port0),
        .out_port1(ofm_port1),
        .port0_valid(ofm_port0_v),
        .port1_valid(ofm_port1_v),
        .start_conv(start_again),
        .odd_cnt(odd_cnt)
    );
    
    /// Buffer
    PSUM_BUFF #(
        .data_width(out_data_width),
        .addr_width(buf_addr_width),
        .depth(buf_depth)
    ) psum_buff0 (
        .clk(clk),
        .rst_n(rst_n),
        .p_valid_data(p_valid_data),
        .p_write_zero(p_write_zero[0]),
        .p_init(p_init),
        .odd_cnt(odd_cnt),
        .pe0_data(pe00_data),
        .pe1_data(pe10_data),
        .pe2_data(pe20_data),
        .pe3_data(pe30_data),
        .fifo_out(fifo_out[0]),
        .valid_fifo_out(valid_fifo_out[0])
    );

    PSUM_BUFF #(
        .data_width(out_data_width),
        .addr_width(buf_addr_width),
        .depth(buf_depth)
    ) psum_buff1 (
        .clk(clk),
        .rst_n(rst_n),
        .p_valid_data(p_valid_data),
        .p_write_zero(p_write_zero[1]),
        .p_init(p_init),
        .odd_cnt(odd_cnt),
        .pe0_data(pe01_data),
        .pe1_data(pe11_data),
        .pe2_data(pe21_data),
        .pe3_data(pe31_data),
        .fifo_out(fifo_out[1]),
        .valid_fifo_out(valid_fifo_out[1])
    );

    PSUM_BUFF #(
        .data_width(out_data_width),
        .addr_width(buf_addr_width),
        .depth(buf_depth)
    ) psum_buff2 (
        .clk(clk),
        .rst_n(rst_n),
        .p_valid_data(p_valid_data),
        .p_write_zero(p_write_zero[2]),
        .p_init(p_init),
        .odd_cnt(odd_cnt),
        .pe0_data(pe02_data),
        .pe1_data(pe12_data),
        .pe2_data(pe22_data),
        .pe3_data(pe32_data),
        .fifo_out(fifo_out[2]),
        .valid_fifo_out(valid_fifo_out[2])
    );

    PSUM_BUFF #(
        .data_width(out_data_width),
        .addr_width(buf_addr_width),
        .depth(buf_depth)
    ) psum_buff3 (
        .clk(clk),
        .rst_n(rst_n),
        .p_valid_data(p_valid_data),
        .p_write_zero(p_write_zero[3]),
        .p_init(p_init),
        .odd_cnt(odd_cnt),
        .pe0_data(pe03_data),
        .pe1_data(pe13_data),
        .pe2_data(pe23_data),
        .pe3_data(pe33_data),
        .fifo_out(fifo_out[3]),
        .valid_fifo_out(valid_fifo_out[3])
    );

    PSUM_BUFF #(
        .data_width(out_data_width),
        .addr_width(buf_addr_width),
        .depth(buf_depth)
    ) psum_buff4 (
        .clk(clk),
        .rst_n(rst_n),
        .p_valid_data(p_valid_data),
        .p_write_zero(p_write_zero[4]),
        .p_init(p_init),
        .odd_cnt(odd_cnt),
        .pe0_data(pe04_data),
        .pe1_data(pe14_data),
        .pe2_data(pe24_data),
        .pe3_data(pe34_data),
        .fifo_out(fifo_out[4]),
        .valid_fifo_out(valid_fifo_out[4])
    );

endmodule // CONV_ACC

module PE_FSM (clk, rst_n, start_conv, start_again, cfg_ci, cfg_co, 
            ifm_read, wgt_read, p_valid_output, last_chanel_output, end_conv
);

    input clk;
    input rst_n;
    input start_conv;
    input start_again;
    input [1:0] cfg_ci;
    input [1:0] cfg_co;
    output ifm_read;
    output wgt_read;
    output p_valid_output;
    output last_chanel_output;
    output end_conv;


    reg [5:0] ci;
    reg [5:0] co;

    reg [5:0] cnt1;
    reg [8:0] cnt2;
    reg [4:0] cnt3;


    reg [2:0] current_state;
    reg [2:0] next_state;

    reg ifm_read;
    reg wgt_read;
    reg p_valid;
    reg last_chanel;
    reg end_conv;

    parameter [2:0]
        IDLE = 3'b000,
        S1 = 3'b001,
        S2 = 3'b010,
        FINISH = 3'b100; 

    parameter[6:0] tile_length =  16;

    always @ (posedge clk or negedge rst_n)
        if(!rst_n)
            current_state <= IDLE;
        else
            current_state <= next_state; 

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ci <= 0;
            co <= 0;
        end else if(start_conv) begin
            ci <= ((cfg_ci + 6'b000001) << 3);
            co <= ((cfg_co + 6'b000001) << 3);
        end
    end

    always @ (current_state or start_conv or start_again or cnt1 or cnt2) 
    begin
        next_state = 3'bx; 
        case(current_state)
            IDLE: 
                if(start_again)
                    next_state = S1;
                else if(start_again && cnt2 == 0 && cnt3 == 0)
                    next_state = FINISH;
                else
                    next_state = IDLE;
            S1: 
                next_state = (cnt1 == 4) ? S2 : S1;
            S2:
                if(cnt2 == 0 && cnt1 == 0)
                    next_state = IDLE;
                else if(cnt1 == 0)
                    next_state = S1;
                // else if(cnt1 == 0)
                //     next_state = S1;
                else 
                    next_state = S2;
            default:
                next_state = IDLE;
        endcase
    end

    always @ (posedge clk or negedge rst_n)
        if(!rst_n)
            {ifm_read, wgt_read, p_valid, last_chanel, end_conv} <= 5'b00000;
        else
            begin
                {ifm_read, wgt_read, p_valid, last_chanel, end_conv} <= 5'b00000;
                case(next_state)
                    IDLE:
                        {ifm_read, wgt_read, p_valid, last_chanel, end_conv} <= 5'b00000;
                    S1: 
                    begin
                        {ifm_read, wgt_read, end_conv} <= 3'b110;
                        p_valid <= (cnt1 < 3) ? 0 : 1; 
                        last_chanel <= (cnt1 == 3 && cnt2 == 0) ? 1 : 0; 
                    end
                    S2:
                        begin
                            {ifm_read, wgt_read, p_valid, end_conv} <= 4'b1010; 
                            last_chanel <= (cnt2 == 0) ? 1 : 0;               
                        end
                    FINISH:
                        {ifm_read, wgt_read, p_valid, last_chanel, end_conv} <= 5'b00001;           
                    default:
                        {ifm_read, wgt_read, p_valid, last_chanel, end_conv} <= 5'b00000;
                endcase
            end



    always @ (posedge clk or negedge rst_n)
        if(!rst_n)
            begin
                cnt1 <= 0;
                cnt2 <= 0;
                cnt3 <= 0;
            end
        else
            begin
                if(next_state == IDLE)
                    cnt1 <= 0;
                else
                    begin
                        if (cnt1 == tile_length + 2)
                            cnt1 <= 0;
                        else      
                            cnt1 <= cnt1 + 1;
                        if(cnt1 == 0)
                        begin                     
                            if(cnt2 == ci-1)
                                cnt2 <= 0;
                            else
                                cnt2 <= cnt2 + 1;
                            if(cnt2 == 0)
                                if(cnt3 == co*15-1)
                                    cnt3 <= 0;
                                else
                                    cnt3 <= cnt3 + 1;
                            else 
                                cnt3 <= cnt3;
                        end
                        else
                            cnt2 <= cnt2;
                    end
            end

    reg [2:0] p_valid_i;
    reg [2:0] last_chanel_i;
    reg p_valid_output;
    reg last_chanel_output;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            p_valid_output     <= 0;
            p_valid_i[2]       <= 0;
            p_valid_i[1]       <= 0;
            p_valid_i[0]       <= 0;
            last_chanel_output <= 0; 
            last_chanel_i[2]   <= 0;
            last_chanel_i[1]   <= 0;
            last_chanel_i[0]   <= 0;
        end else begin
            p_valid_output     <= p_valid_i[2];
            p_valid_i[2]       <= p_valid_i[1];
            p_valid_i[1]       <= p_valid_i[0];
            p_valid_i[0]       <= p_valid;
            last_chanel_output <= last_chanel_i[2]; 
            last_chanel_i[2]   <= last_chanel_i[1];
            last_chanel_i[1]   <= last_chanel_i[0];
            last_chanel_i[0]   <= last_chanel;
        end

endmodule // PE_FSM

module WGT_BUF (clk, rst_n, wgt_input, wgt_read, wgt_buf0, wgt_buf1, wgt_buf2, wgt_buf3);

    input clk;
    input rst_n;
    input signed [7:0] wgt_input;
    input wgt_read;
    output signed [7:0] wgt_buf0;
    output signed [7:0] wgt_buf1;
    output signed [7:0] wgt_buf2;
    output signed [7:0] wgt_buf3;

    reg signed [7:0] wgt_buf [3:0];


    integer i;

    always @(posedge clk or negedge rst_n) 
        if (~rst_n) 
        begin
            for(i = 0; i < 4; i = i + 1) 
            begin
                wgt_buf[i] <= 0;
            end
        end
        else
        begin
            if(wgt_read)
            begin
                wgt_buf[3] <= wgt_buf[2];
                wgt_buf[2] <= wgt_buf[1];
                wgt_buf[1] <= wgt_buf[0];
                wgt_buf[0] <= wgt_input;
            end
            else
            begin
                wgt_buf[3] <= wgt_buf[3];
                wgt_buf[2] <= wgt_buf[2];
                wgt_buf[1] <= wgt_buf[1];
                wgt_buf[0] <= wgt_buf[0];
            end
        end

        assign wgt_buf0 = wgt_buf[0];
        assign wgt_buf1 = wgt_buf[1];
        assign wgt_buf2 = wgt_buf[2];
        assign wgt_buf3 = wgt_buf[3];

endmodule 

module IFM_BUF (clk, rst_n, ifm_input, ifm_read, ifm_buf0, ifm_buf1, ifm_buf2, ifm_buf3);

    input clk;
    input rst_n;
    input signed [7:0] ifm_input;
    input ifm_read;
    output signed [7:0] ifm_buf0;
    output signed [7:0] ifm_buf1;
    output signed [7:0] ifm_buf2;
    output signed [7:0] ifm_buf3;

    reg signed [7:0] ifm_buf [3:0];

    integer i;

    always @(posedge clk or negedge rst_n) 
        if (~rst_n) 
        begin
            for(i = 0; i < 4; i = i + 1) 
            begin
                ifm_buf[i] <= 0;
            end
        end
        else
        begin
            if(ifm_read)
            begin
                ifm_buf[3] <= ifm_buf[2];
                ifm_buf[2] <= ifm_buf[1];
                ifm_buf[1] <= ifm_buf[0];
                ifm_buf[0] <= ifm_input;
            end
            else 
            begin
                ifm_buf[3] <= ifm_buf[3];
                ifm_buf[2] <= ifm_buf[2];
                ifm_buf[1] <= ifm_buf[1];
                ifm_buf[0] <= ifm_buf[0];
            end
        end

    assign ifm_buf0 = ifm_buf[0];
    assign ifm_buf1 = ifm_buf[1];
    assign ifm_buf2 = ifm_buf[2];
    assign ifm_buf3 = ifm_buf[3];

endmodule // IFM_BUF

module PE (clk, rst_n, ifm_input0, ifm_input1, ifm_input2, ifm_input3, 
            wgt_input0, wgt_input1, wgt_input2, wgt_input3, p_sum);

    input clk;
    input rst_n;
    input signed [7:0] ifm_input0;
    input signed [7:0] ifm_input1;
    input signed [7:0] ifm_input2;
    input signed [7:0] ifm_input3;

    input signed [7:0] wgt_input0;
    input signed [7:0] wgt_input1;
    input signed [7:0] wgt_input2;
    input signed [7:0] wgt_input3;

    output signed [24:0] p_sum;

    reg signed [15:0] product [3:0];
    reg signed [16:0] pp_sum [1:0];
    reg signed [24:0] p_sum;


    integer i;
    integer j;

    always @(posedge clk or negedge rst_n) 
        if (~rst_n) 
        begin
            for(i = 0; i < 4; i = i + 1) 
            begin
                product[i] <= 0;
            end

            for(i = 0; i < 2; i = i + 1) 
            begin
                pp_sum[i] <= 0;
            end
            p_sum <= 0;
        end
        else
        begin
            product[0] <= ifm_input0 * wgt_input0;
            product[1] <= ifm_input1 * wgt_input1;
            product[2] <= ifm_input2 * wgt_input2;
            product[3] <= ifm_input3 * wgt_input3;

            pp_sum[0] <= product[0] + product[1];
            pp_sum[1] <= product[2] + product[3];
            p_sum <= pp_sum[0] + pp_sum[1];
        end

endmodule // PE

///==------------------------------------------------------------------==///
/// Conv kernel: writeback controller
///==------------------------------------------------------------------==///
module WRITE_BACK #(
    parameter data_width = 25,
    parameter depth = 61
) (
    input  clk,
    input  rst_n,
    input  start_init,
    input  p_filter_end,
    input  [data_width-1:0] row0,
    input  row0_valid,
    input  [data_width-1:0] row1,
    input  row1_valid,
    input  [data_width-1:0] row2,
    input  row2_valid,
    input  [data_width-1:0] row3,
    input  row3_valid,
    input  [data_width-1:0] row4,
    input  row4_valid,
    output p_write_zero0,
    output p_write_zero1,
    output p_write_zero2,
    output p_write_zero3,
    output p_write_zero4,
    output p_init,
    output [data_width-1:0] out_port0,
    output [data_width-1:0] out_port1,
    output port0_valid,
    output port1_valid,
    output start_conv,
    output odd_cnt
);
    /// machine state encode
    localparam IDLE         = 4'd0;
    localparam INIT_BUFF    = 4'd1;
    localparam START_CONV   = 4'd2;
    localparam WAIT_ADD     = 4'd3;
    localparam WAIT_WRITE0    = 4'd4;
    localparam ROW_0_1      = 4'd5;
    localparam CLEAR_0_1    = 4'd6;
    localparam ROW_2_3      = 4'd7;
    localparam CLEAR_2_3    = 4'd8;
    localparam ROW_5        = 4'd9;
    localparam CLEAR_START_CONV = 4'd10;
    localparam CLEAR_CNT    = 4'd11;

    // localparam DONE         = 4'b1001;
    /// machine state
    reg [3:0] st_next;
    reg [3:0] st_cur;
    reg [7:0] cnt;
    /// State transfer
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            st_cur <= IDLE;
        else 
            st_cur <= st_next;
    end
    /// Next state logic
    always @(*) begin
        st_next = st_cur;
        case(st_cur)
            IDLE:
                if (start_init)
                    st_next = INIT_BUFF;
                else
                    st_next = IDLE;
            INIT_BUFF:
                if (cnt == depth-1)
                    st_next = START_CONV;
                else
                    st_next = INIT_BUFF;
            START_CONV:
                if (cnt >= depth+2)
                    st_next = CLEAR_START_CONV;
                else 
                    st_next = START_CONV;
            CLEAR_START_CONV:
                if (p_filter_end)
                    st_next = WAIT_ADD;
                else
                    st_next = CLEAR_START_CONV;
            WAIT_ADD:
                if (cnt == depth-1)
                    st_next = WAIT_WRITE0;
                else
                    st_next = WAIT_ADD;
            WAIT_WRITE0:
                st_next = CLEAR_CNT;
            CLEAR_CNT:
                st_next = ROW_0_1;
            ROW_0_1:
                if (cnt == depth-1)
                    st_next = CLEAR_0_1;
                else
                    st_next = ROW_0_1;
            CLEAR_0_1:
                st_next = ROW_2_3;
            ROW_2_3:
                if (cnt == depth-1)
                    st_next = CLEAR_2_3;
                else
                    st_next = ROW_2_3;
            CLEAR_2_3:
                st_next = ROW_5;
            ROW_5:
                if (cnt == depth-1)
                    st_next = CLEAR_START_CONV;
                else
                    st_next = ROW_5;
            // DONE:
            //     st_next = START_CONV;
            default:
                st_next = IDLE;  
        endcase
    end
    /// Output logic
    reg p_write_zero0_r;
    reg p_write_zero1_r;
    reg p_write_zero2_r;
    reg p_write_zero3_r;
    reg p_write_zero4_r;
    reg p_init_r;
    reg [data_width-1:0] out_port0_r;
    reg [data_width-1:0] out_port1_r;
    reg port0_valid_r;
    reg port1_valid_r;
    reg start_conv_r;
    /// Output start conv signal
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            start_conv_r <= 0;
        else if (st_cur == START_CONV || st_cur == CLEAR_CNT)
            start_conv_r <= 1;
        else
            start_conv_r <= 0;
    end
    assign start_conv = start_conv_r;
    /// PingPong buffer controller signal
    reg odd_cnt_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            odd_cnt_r <= 0;
        else if (st_cur == CLEAR_CNT)
            odd_cnt_r <= ~odd_cnt;
        else
            odd_cnt_r <= odd_cnt;
    end
    assign odd_cnt = odd_cnt_r;
    /// Output zero flag signals
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_write_zero0_r <= 0;
            p_write_zero1_r <= 0;
        end else if (st_cur == ROW_0_1) begin
            p_write_zero0_r <= 1;  
            p_write_zero1_r <= 1;
        end else begin
            p_write_zero0_r <= 0;
            p_write_zero1_r <= 0;
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_write_zero2_r <= 0;
            p_write_zero3_r <= 0;
        end else if (st_cur == ROW_2_3) begin
            p_write_zero2_r <= 1;  
            p_write_zero3_r <= 1;
        end else begin
            p_write_zero2_r <= 0;
            p_write_zero3_r <= 0;
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p_write_zero4_r <= 0;
        end else if (st_cur == ROW_5) begin
            p_write_zero4_r <= 1;  
        end else begin
            p_write_zero4_r <= 0;
        end
    end
    assign p_write_zero0 = p_write_zero0_r;
    assign p_write_zero1 = p_write_zero1_r;
    assign p_write_zero2 = p_write_zero2_r;
    assign p_write_zero3 = p_write_zero3_r;
    assign p_write_zero4 = p_write_zero4_r;   
    /// Init buffer signal, why this signal? since, at the beginning, the buffer is empty, we only need to
    /// push zero to buffer without read from it, this behaviour is difference from p_write_zerox signals
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            p_init_r <= 0;
        else if (st_cur == INIT_BUFF)
            p_init_r <= 1;
        else
            p_init_r <= 0;
    end
    assign p_init = p_init_r;
    /// Update the cnt
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 0;
        else if (st_cur == IDLE || st_cur == CLEAR_0_1  || st_cur == CLEAR_START_CONV
            || st_cur == CLEAR_2_3 || st_cur == CLEAR_CNT)
            cnt <= 0;
        else 
            cnt <= cnt + 1;
    end
    /// Final result, a big mux
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_port0_r <= 0;
            out_port1_r <= 0;
            port0_valid_r <= 0;
            port1_valid_r <= 0;
        end else begin
            case({row0_valid, row1_valid, row2_valid, row3_valid, row4_valid})
                5'b11000 : begin
                    out_port0_r <= row0;
                    out_port1_r <= row1;
                    port0_valid_r <= row0_valid;
                    port1_valid_r <= row1_valid;
                end
                5'b00110 : begin
                    out_port0_r <= row2;
                    out_port1_r <= row3;
                    port0_valid_r <= row2_valid;
                    port1_valid_r <= row3_valid;
                end
                5'b00001 : begin
                    out_port0_r <= row4;
                    out_port1_r <= 0;
                    port0_valid_r <= row4_valid;
                    port1_valid_r <= 0;
                end
                default : begin
                    out_port0_r <= 0;
                    out_port1_r <= 0;
                    port0_valid_r <= 0;
                    port1_valid_r <= 0;
                end    
            endcase
        end
    end
    assign out_port0 = out_port0_r;
    assign out_port1 = out_port1_r;
    assign port0_valid = port0_valid_r;
    assign port1_valid = port1_valid_r;
endmodule

///==------------------------------------------------------------------==///
/// Conv kernel: synchronous FIFO
///==------------------------------------------------------------------==///
/// Synchronous FIFO
module SYNCH_FIFO #(
    parameter data_width = 25,
    parameter addr_width = 8,
    parameter depth      = 61
) (
    /// Control signal
    input clk,
    input rd_en,
    input wr_en,
    input rst_n,
    /// status signal
    output empty,
    output full,
    /// data signal
    output reg [data_width-1:0] data_out,
    input [data_width-1:0] data_in
);
    reg [addr_width:0] cnt;
    reg [data_width-1:0] fifo_mem [0:depth-1];
    reg [addr_width-1:0] rd_ptr;
    reg [addr_width-1:0] wr_ptr;
    /// Status generation
    assign empty = (cnt == 0);
    assign full  = (cnt == depth);
    /// Updata read pointer && Read operation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_ptr   <= 0;
        else if (rd_en && !empty) begin
            if (rd_ptr == depth-1)
                rd_ptr <= 0;
            else
                rd_ptr <= rd_ptr + 1;
        end else 
            rd_ptr <= rd_ptr;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) 
            data_out <= 0;
        else if (rd_en && !empty)
            data_out <= fifo_mem[rd_ptr];
    end
    /// Update write pointer && write operation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wr_ptr <= 0;
        else if (wr_en  && !full) begin
            if (wr_ptr == depth-1)
                wr_ptr <= 0;
            else
                wr_ptr <= wr_ptr + 1;
        end else 
            wr_ptr <= wr_ptr;
    end

    always @(posedge clk) begin
        if (wr_en  & ~full)
            fifo_mem[wr_ptr] = data_in;
    end
    /// Update the counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cnt <= 0;
        else begin
            case ({wr_en, rd_en})
                2'b00: cnt <= cnt;
                2'b01: cnt <= !empty ? cnt-1 : cnt;
                2'b10: cnt <= !full  ? cnt+1 : cnt;
                2'b11: cnt <= cnt;
            endcase
        end
    end
endmodule

///==------------------------------------------------------------------==///
/// Conv kernel: partial sum buffer
///==------------------------------------------------------------------==///
module PSUM_BUFF #(
    parameter data_width = 25,
    parameter addr_width = 8,
    parameter depth      = 61
) (
    input clk,
    input rst_n,
    input p_valid_data,
    input p_write_zero,
    input p_init,
    input odd_cnt,
    input signed [data_width-1:0] pe0_data,
    input signed [data_width-1:0] pe1_data,
    input signed [data_width-1:0] pe2_data,
    input signed [data_width-1:0] pe3_data,
    output [data_width-1:0] fifo_out,
    output valid_fifo_out
);
    // wire [data_width-1:0] fifo_head;
    // reg  [data_width-1:0] fifo_head_reg;
    reg  [data_width-1:0] fifo_in0;
    reg  [data_width-1:0] fifo_in1;

    wire signed [data_width-1:0] adder_out;
    wire empty0, full0;
    wire empty1, full1;

    reg fifo_rd_en0;
    reg fifo_wr_en0;
    reg fifo_rd_en1;
    reg fifo_wr_en1;
    /// delayed odd_cnt
    reg d_odd_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            d_odd_cnt <= 0;
        else 
            d_odd_cnt <= odd_cnt;
    end

    /// Shifter register, there are four pipeline stages
    reg [3-1:0] p_valid;
    reg [3-1:0] p_write_zero_reg;

    reg [data_width-1:0] fifo_out_i;
    reg [data_width-1:0] fifo_out_a;
    wire [data_width-1:0] fifo_out_i0;
    wire [data_width-1:0] fifo_out_i1;

    always @(*) begin
        if (!rst_n) begin
            fifo_out_i = 0;
            fifo_out_a = 0;
        end else if (~d_odd_cnt) begin
            fifo_out_i = fifo_out_i1;
            fifo_out_a = fifo_out_i0;
        end else begin
            fifo_out_i = fifo_out_i0;
            fifo_out_a = fifo_out_i1;
        end
    end

    /// Whether the output of current fifo output is valid
    assign valid_fifo_out = p_write_zero_reg;

    /// Relu Operation
    assign fifo_out = fifo_out_i[data_width-1] ? 0 : fifo_out_i;

    /// Fifo read and write
    /// When to write fifo
    /// Data that will be written to fifo
    wire write_zero;
    assign write_zero = p_write_zero || p_write_zero_reg;
    always @(*) begin
        if (!rst_n) begin
            fifo_rd_en0 = 0;
            fifo_wr_en0 = 0;
            fifo_rd_en1 = 0;
            fifo_wr_en1 = 0;
            fifo_in0 = 0;
            fifo_in1 = 0;
        end else if (p_init) begin
            fifo_rd_en0 = 0;
            fifo_wr_en0 = 1;
            fifo_rd_en1 = 0;
            fifo_wr_en1 = 1;
            fifo_in0 = 0;
            fifo_in1 = 0;
        end else if (write_zero & d_odd_cnt) begin
            fifo_rd_en0 = p_write_zero;
            fifo_wr_en0 = p_write_zero_reg;
            fifo_rd_en1 = p_valid[0];
            fifo_wr_en1 = p_valid[2];
            fifo_in0 = 0;
            fifo_in1 = adder_out;
        end else if (write_zero & ~d_odd_cnt) begin
            fifo_rd_en1 = p_write_zero;
            fifo_wr_en1 = p_write_zero_reg;
            fifo_rd_en0 = p_valid[0];
            fifo_wr_en0 = p_valid[2];
            fifo_in1 = 0;
            fifo_in0 = adder_out;
        end else if (~d_odd_cnt & ~write_zero) begin
            fifo_rd_en1 = 0;
            fifo_wr_en1 = 0;
            fifo_rd_en0 = p_valid[0];
            fifo_wr_en0 = p_valid[2];
            fifo_in1 = 0;
            fifo_in0 = adder_out;
        end else if (d_odd_cnt & ~write_zero) begin
            fifo_rd_en0 = 0;
            fifo_wr_en0 = 0;
            fifo_rd_en1 = p_valid[0];
            fifo_wr_en1 = p_valid[2];
            fifo_in0 = 0;
            fifo_in1 = adder_out;
        end else begin
            fifo_rd_en0 = 0;
            fifo_wr_en0 = 0;
            fifo_rd_en1 = 0;
            fifo_wr_en1 = 0;
            fifo_in0 = 0;
            fifo_in1 = 0;
        end
    end
    /// Whether the current input is valid data
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            p_valid <= 3'b0;
        else 
            p_valid <= {p_valid[1:0], p_valid_data};
    end
    /// Whether the current write zero is valid data
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            p_write_zero_reg <= 1'b0;
        else 
            p_write_zero_reg <= p_write_zero;
    end
    /// Adder tree
    PSUM_ADD #(.data_width(data_width)) adder_tree (
        .clk(clk),
        .rst_n(rst_n),
        .pe0_data(pe0_data),
        .pe1_data(pe1_data),
        .pe2_data(pe2_data),
        .pe3_data(pe3_data),
        .fifo_data(fifo_out_a),
        .out(adder_out)
    );
    /// Synchronous fifo
    SYNCH_FIFO #(
        .data_width(data_width),
        .addr_width(addr_width),
        .depth(depth)
    ) synch_fifo0 (
        .clk(clk),
        .rd_en(fifo_rd_en0),
        .wr_en(fifo_wr_en0),
        .rst_n(rst_n),
        .empty(empty0),
        .full(full0),
        .data_out(fifo_out_i0),
        .data_in(fifo_in0)
    );

    /// Synchronous fifo
    SYNCH_FIFO #(
        .data_width(data_width),
        .addr_width(addr_width),
        .depth(depth)
    ) synch_fifo1 (
        .clk(clk),
        .rd_en(fifo_rd_en1),
        .wr_en(fifo_wr_en1),
        .rst_n(rst_n),
        .empty(empty1),
        .full(full1),
        .data_out(fifo_out_i1),
        .data_in(fifo_in1)
    );

endmodule

///==------------------------------------------------------------------==///
/// Conv kernel: adder tree of psum module
///==------------------------------------------------------------------==///
/// Three stages pipelined adder tree
module PSUM_ADD #(
    parameter data_width = 25
) (
    input clk,
    input rst_n,
    input signed [data_width-1:0] pe0_data,
    input signed [data_width-1:0] pe1_data,
    input signed [data_width-1:0] pe2_data,
    input signed [data_width-1:0] pe3_data,
    input signed [data_width-1:0] fifo_data,
    output signed [data_width-1:0] out
);

    reg signed [data_width-1:0] psum0;
    reg signed [data_width-1:0] psum1;
    reg signed [data_width-1:0] psum2;
    reg signed [data_width-1:0] out_r;

    assign out = out_r;
    /// Adder tree
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum0 <= 0;
            psum1 <= 0;
            psum2 <= 0;
            out_r <= 0;
        end else begin
            psum0 <= pe0_data + pe1_data;
            psum1 <= pe2_data + pe3_data;
            psum2 <= psum0 + psum1;
            out_r <= fifo_data + psum2;
        end
    end
endmodule


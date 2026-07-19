// design_clean.v
// A small, clean RS232-like transmitter FSM with an 8-bit data path.
// No hardware Trojan. This is the reference "golden" design.
module rs232_tx (
    input        clk,
    input        rst_n,
    input        send,        // pulse to start a transmission
    input  [7:0] data_in,     // byte to transmit
    output reg   tx,          // serial line out
    output reg   busy         // high while shifting a byte out
);

    // FSM states
    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam SHIFT = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0] state;
    reg [2:0] bit_idx;        // which bit we are shifting (0..7)
    reg [7:0] shreg;          // shift register holding the byte

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            tx      <= 1'b1;  // line idles high
            busy    <= 1'b0;
            bit_idx <= 3'd0;
            shreg   <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    tx   <= 1'b1;
                    busy <= 1'b0;
                    if (send) begin
                        shreg <= data_in;
                        busy  <= 1'b1;
                        state <= START;
                    end
                end
                START: begin
                    tx      <= 1'b0;   // start bit
                    bit_idx <= 3'd0;
                    state   <= SHIFT;
                end
                SHIFT: begin
                    tx <= shreg[bit_idx];
                    if (bit_idx == 3'd7)
                        state <= STOP;
                    else
                        bit_idx <= bit_idx + 3'd1;
                end
                STOP: begin
                    tx    <= 1'b1;     // stop bit
                    busy  <= 1'b0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule

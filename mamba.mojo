
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from random import rand
from algorithm import vectorize
from sys.info import simdwidthof
import math
from python import Python


"""
    B: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    L: sequence length                  (`L` in [1] Algorithm 2)
    D_MODEL: hidden dim
    D_STATE: latent state dim           (`N` in [1] Algorithm 2)
    EXPAND: expansion factor            (`E` in [1] Section 3.4)
    D_INNER: D_STATE * EXPAND           (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
let B = 4
let L = 128
let D_MODEL = 512
let D_STATE = 128
let EXPAND = 4
let D_INNER = D_STATE * EXPAND
alias simd_float32_width: Int = simdwidthof[DType.float32]()

alias floattensor = Tensor[DType.float32]
fn naive_matmul(A: floattensor, B: floattensor) -> floattensor:
    var output = Tensor[DType.float32](A.shape()[0], B.shape()[1])
    for i in range(A.shape()[0]):
        for j in range(B.shape()[1]):
            for k in range(A.shape()[1]):
                output[i][j] += A[i][k] * B[k][j]
    return output

fn elementwise_exp(input: floattensor) -> floattensor:
    var output: floattensor = Tensor[DType.float32](input.shape())
    @parameter
    fn exp_simd[simd_float32_width: Int](idx: Int) -> None:
        output.simd_store[simd_float32_width](idx, math.exp(input.simd_load[simd_float32_width](idx)))
    vectorize[simd_float32_width, exp_simd](input.num_elements())

struct LinearLayer:
    var D_IN: Int
    var D_OUT: Int
    var weights: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    var add_bias: Bool
    
    fn __init__(inout self, in_features: Int, out_features: Int, add_bias: Bool = False):
        self.D_IN = in_features
        self.D_OUT = out_features
        self.weights = rand[DType.float32](self.D_IN, self.D_OUT)
        self.bias = rand[DType.float32](self.D_OUT, 1)
        self.add_bias = add_bias
    
    fn forward(self, input: Tensor[DType.float32]) -> Tensor[DType.float32]:
        let output: Tensor[DType.float32] = naive_matmul(self.weights, input)
        return output

struct Conv1D:
    fn __init__(inout self):
        pass

struct MambaBlock:
    var in_projection: LinearLayer
    var x_projection: LinearLayer
    var dt_projection: LinearLayer
    var out_projection: LinearLayer
    var dt_rank: Int
    var conv1d: Conv1D

    fn __init__(inout self) raises:
        let py_math = Python.import_module("math")
        self.dt_rank = int(py_math.ceil(D_MODEL / 16))
        self.in_projection = LinearLayer(D_MODEL, D_INNER * 2)
        self.x_projection = LinearLayer(D_INNER, self.dt_rank)
        self.dt_projection = LinearLayer(self.dt_rank, D_INNER)
        self.out_projection = LinearLayer(D_INNER, D_MODEL)
        self.conv1d = Conv1D()

    fn state_space_model(self):
        pass

    # Ugly but we'll keep it for now
    fn einsum_bl_din_din_n_to_b_din_l_n(self, delta: Tensor[DType.float32], A: Tensor[DType.float32]) -> Tensor[DType.float32]:
        var b = delta.shape()[0]
        var l = delta.shape()[1]
        var d_in = delta.shape()[2]

        var n = A.shape()[1]
        
        # Initialize output tensor with the desired shape: 'b d_in l n'
        var output = Tensor[DType.float32](b, d_in, l, n)

        # Perform the operation
        for b_index in range(b):
            for l_index in range(l):
                for d_in_index in range(d_in):
                    for n_index in range(n):
                        output[b_index][d_in_index][l_index][n_index] += delta[b_index][l_index][d_in_index] * A[d_in_index][n_index]
        return output

    fn selective_scan(self, u: floattensor, delta: floattensor, 
                      A: floattensor, B: floattensor, C: floattensor, D: floattensor) raises -> floattensor:
        let b = u.shape()[0]
        let l = u.shape()[1]
        let d_in = u.shape()[2]
        let n = A.shape()[1]

        var deltaA: floattensor = elementwise_exp(self.einsum_bl_din_din_n_to_b_din_l_n(delta, A)) 
        var deltaB_u: floattensor

        var x = Tensor[DType.float32](b, d_in, n)
        var ys

fn run() -> Int:
    return 0

def main():
    run()
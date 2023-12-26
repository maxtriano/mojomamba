
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

alias floattensor = Tensor[DType.float32]
fn naive_matmul(A: floattensor, B: floattensor) -> floattensor:
    var output = Tensor[DType.float32](A.shape()[0], B.shape()[1])
    for i in range(A.shape()[0]):
        for j in range(B.shape()[1]):
            for k in range(A.shape()[1]):
                output[i][j] += A[i][k] * B[k][j]
    return output

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

    fn selective_scan(self):
        pass

fn run() -> Int:
    return 0

def main():
    run()
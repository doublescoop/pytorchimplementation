# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments.

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py
        
        
        
        
* Sentiment 

Epoch 1, loss 31.49080265986212, train accuracy: 49.33%
Validation accuracy: 48.00%
Best Valid accuracy: 48.00%
Epoch 2, loss 31.368454735359627, train accuracy: 50.67%
Validation accuracy: 48.00%
Best Valid accuracy: 48.00%
Epoch 3, loss 31.048982921999766, train accuracy: 50.00%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 4, loss 30.926031439581656, train accuracy: 54.44%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 5, loss 30.756206922309318, train accuracy: 55.56%
Validation accuracy: 65.00%
Best Valid accuracy: 65.00%
Epoch 6, loss 30.520996736954473, train accuracy: 59.56%
Validation accuracy: 51.00%
Best Valid accuracy: 65.00%
Epoch 7, loss 30.35885425614405, train accuracy: 58.89%
Validation accuracy: 53.00%
Best Valid accuracy: 65.00%
Epoch 8, loss 29.862152644552882, train accuracy: 62.89%
Validation accuracy: 55.00%
Best Valid accuracy: 65.00%
Epoch 9, loss 29.75375495126058, train accuracy: 63.78%
Validation accuracy: 56.00%
Best Valid accuracy: 65.00%
Epoch 10, loss 29.299430011359114, train accuracy: 64.44%
Validation accuracy: 63.00%
Best Valid accuracy: 65.00%
Epoch 11, loss 29.09203274960091, train accuracy: 67.33%
Validation accuracy: 52.00%
Best Valid accuracy: 65.00%
Epoch 12, loss 28.538095610899383, train accuracy: 67.11%
Validation accuracy: 55.00%
Best Valid accuracy: 65.00%
Epoch 13, loss 28.243942566480257, train accuracy: 67.56%
Validation accuracy: 62.00%
Best Valid accuracy: 65.00%
Epoch 14, loss 27.839221666491827, train accuracy: 68.67%
Validation accuracy: 62.00%
Best Valid accuracy: 65.00%
Epoch 15, loss 27.130555843873374, train accuracy: 71.78%
Validation accuracy: 61.00%
Best Valid accuracy: 65.00%
Epoch 16, loss 26.755382481817076, train accuracy: 72.44%
Validation accuracy: 59.00%
Best Valid accuracy: 65.00%
Epoch 17, loss 25.672714597942498, train accuracy: 75.11%
Validation accuracy: 58.00%
Best Valid accuracy: 65.00%
Epoch 18, loss 25.597033456002148, train accuracy: 72.00%
Validation accuracy: 68.00%
Best Valid accuracy: 68.00%
Epoch 19, loss 24.08179607533272, train accuracy: 77.11%
Validation accuracy: 59.00%
Best Valid accuracy: 68.00%
Epoch 20, loss 24.377069071864483, train accuracy: 73.11%
Validation accuracy: 63.00%
Best Valid accuracy: 68.00%
Epoch 21, loss 23.268298229246003, train accuracy: 75.33%
Validation accuracy: 63.00%
Best Valid accuracy: 68.00%
Epoch 22, loss 22.826964035150567, train accuracy: 75.78%
Validation accuracy: 63.00%
Best Valid accuracy: 68.00%
Epoch 23, loss 22.547555289464608, train accuracy: 74.22%
Validation accuracy: 61.00%
Best Valid accuracy: 68.00%
Epoch 24, loss 21.629458565181142, train accuracy: 77.78%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 25, loss 20.794669269833456, train accuracy: 78.44%
Validation accuracy: 67.00%
Best Valid accuracy: 68.00%
Epoch 26, loss 20.474026342109305, train accuracy: 79.56%
Validation accuracy: 64.00%
Best Valid accuracy: 68.00%
Epoch 27, loss 19.756648505195763, train accuracy: 76.67%
Validation accuracy: 62.00%
Best Valid accuracy: 68.00%
Epoch 28, loss 19.24081873422797, train accuracy: 80.22%
Validation accuracy: 71.00%
Best Valid accuracy: 71.00%
Epoch 29, loss 17.950741531449655, train accuracy: 82.89%
Validation accuracy: 69.00%
Best Valid accuracy: 71.00%


* MNIST

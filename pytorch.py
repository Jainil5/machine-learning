import torch 

# Scalar
scalar = torch.tensor(7)

# Get the Python number within a tensor (only works with one-element tensors)
#scalar.item()

# Vector
vector = torch.tensor([7, 7])


# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])

# Check number of dimensions


# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

print(scalar)
print(vector)
print(MATRIX)
print(TENSOR)
print("Shape:")
print(scalar.shape)
print(vector.shape)
print(MATRIX.shape)
print(TENSOR.shape)
print("Ndim")
print(scalar.ndim)
print(vector.ndim)
print(MATRIX.ndim)
print(TENSOR.ndim)

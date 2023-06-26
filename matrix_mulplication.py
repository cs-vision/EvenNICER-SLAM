import numpy as np
import torch


matrix1 = np.array([[-3.2057e-01, -4.4806e-01,  8.3455e-01,  3.4530e+00],
        [ 9.4722e-01, -1.5164e-01,  2.8244e-01,  4.5461e-01],
        [ 1.0790e-16,  8.8105e-01,  4.7302e-01,  5.9363e-01],
        [0,0,0,1]])
# matrix2 = np.array([[ 9.9966e-01, -2.6146e-02, -3.5900e-04, -6.7127e-04],
#         [ 2.6141e-02,  9.9961e-01, -1.0219e-02, -5.0680e-04],
#         [ 6.2605e-04,  1.0206e-02,  9.9995e-01,  5.2363e-04]])

matrix2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]) #(3,4)
matrix2 = torch.tensor(matrix2, dtype=torch.float)

bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32)
matrix2 = torch.cat([matrix2, bottom], dim=0)

matrix1 = torch.tensor(matrix1[:3, :], dtype=torch.float)  # (3,4)
matrix2 = torch.tensor(matrix2, dtype=torch.float)

result = torch.matmul(matrix2,matrix1.t()) #(4,4)*(4,3)=(4,3)


print(result.t()) # (3,4)
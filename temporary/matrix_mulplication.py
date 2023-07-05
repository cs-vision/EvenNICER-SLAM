import numpy as np
import torch
from src.common import get_camera_from_tensor

tensor = torch.tensor([[0, 0, 0, 0.94, -2.5, -0.3, 0.45]]).to("cuda:0")
matrix1 = get_camera_from_tensor(tensor).cpu()


# matrix1 = np.array([[-3.2057e-01, -4.4806e-01,  8.3455e-01,  3.4530e+00],
#         [ 9.4722e-01, -1.5164e-01,  2.8244e-01,  4.5461e-01],
#         [ 1.0790e-16,  8.8105e-01,  4.7302e-01,  5.9363e-01],
#         [0,0,0,1]])
matrix2 = np.array([[ 0.9995, -0.0243, 0.0207, -0.0013],
        [ 0.0245,  0.9996, -0.0104, -5.0680e-04],
        [ -0.02,  0.0109,  9.9995e-01,  -0.0044]])

#matrix2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1]]) #(3,4)
matrix2 = torch.tensor(matrix2, dtype=torch.float)

bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32)
print(matrix1.shape)
#matrix1 = torch.cat([matrix1, bottom], dim=0)
matrix2 = torch.cat([matrix2, bottom], dim=0)

matrix1 = torch.tensor(matrix1, dtype=torch.float)  # (3,4)
#print(matrix1.t())
#matrix2 = torch.tensor(matrix2, dtype=torch.float)

result = torch.matmul(matrix1,matrix2)[:3, :] #(4,4)*(4,3)=(4,3)
print(result)
#result = torch.matmul(matrix2,matrix1.t()) #(4,4)*(4,3)=(4,3)

#print(result)

#print(result.t()) # (3,4)
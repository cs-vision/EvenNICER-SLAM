import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion

def get_tensor_from_camera_in_pytorch(RT, Tquad=False):
    N = len(RT.shape)
    print(N)
    if N == 2:
        RT = RT.unsqueeze(0)
    R, T = RT[:, :3, :3], RT[:, :3, 3]
    quaternion = matrix_to_quaternion(R)
    if Tquad:
        tensor = torch.cat((T, quaternion), 1)
    else:
        tensor = torch.cat((quaternion, T), 1)
    if N == 2:
        tensor = tensor[0]
    return tensor

matrix1 = np.array([[[-3.2057e-01, -4.4806e-01,  8.3455e-01,  3.4530e+00],
        [ 9.4722e-01, -1.5164e-01,  2.8244e-01,  4.5461e-01],
        [ 1.0790e-16,  8.8105e-01,  4.7302e-01,  5.9363e-01],
        [0,0,0,1]],
        [[-3.2057e-01, -4.4806e-01,  8.3455e-01,  3.4530e+00],
        [ 9.4722e-01, -1.5164e-01,  2.8244e-01,  4.5461e-01],
        [ 1.0790e-16,  8.8105e-01,  4.7302e-01,  5.9363e-01],
        [0,0,0,1]]])
#print(matrix1.shape)
matrix2_1 = np.array([[ 9.9966e-01, -2.6146e-02, -3.5900e-04, -6.7127e-04],
        [ 2.6141e-02,  9.9961e-01, -1.0219e-02, -5.0680e-04],
        [ 6.2605e-04,  1.0206e-02,  9.9995e-01,  5.2363e-04]])

matrix2_1 = torch.tensor(matrix2_1, dtype=torch.float)
print(matrix2_1.shape)

matrix2 = np.array([[[1,0,0,0],[0,1,0,0],[0,0,1,1]],
                   [[1, 0, 0, 0], [0, 1, 0, 0],[1, 0, 0, 0]]]) #(2,3,4)
matrix2 = torch.tensor(matrix2, dtype=torch.float)
#print(matrix2.shape)

additional_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
additional_rows = torch.stack([additional_row] * 2)
matrix2 = torch.cat((matrix2, additional_rows.unsqueeze(1)), dim=1)
#print(matrix2.shape)



#bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32)
#matrix2 = torch.cat([matrix2, bottom], dim=0)

matrix1 = torch.tensor(matrix1, dtype=torch.float)  
#print(matrix1.t())
#matrix2 = torch.tensor(matrix2, dtype=torch.float)

# 2*4*4, 2*4*4

result = torch.matmul(matrix1,matrix2)[:, :3, :] #(4,4)*(4,3)=(4,3)
#print(result.shape)

tensor = get_tensor_from_camera_in_pytorch(result)
print(tensor.shape)


#result = torch.matmul(matrix2,matrix1.t()) #(4,4)*(4,3)=(4,3)

#print(result)

#print(result.t()) # (3,4)
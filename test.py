from knn_cuda import KNN
import torch


knn = KNN(
    k=1,
    transpose_mode=True
)


a=torch.randn(1,1000,3).cuda()
b=torch.randn(1,1000,3).cuda()


dist,idx=knn(a,b)


print(dist.shape)
print(idx.shape)
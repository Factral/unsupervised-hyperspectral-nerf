import torch

mse2psnr = lambda x : -10. * torch.log(x) / torch.log( torch.Tensor([10.]).to(x.device) )

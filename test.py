import torch
from torchvision.models import resnet18


model = resnet18(pretrained=True)
model.eval()

batchsize = 512
x = torch.randn(batchsize, 3, 224, 224)

logits = []
for i in range(batchsize):
    x_i = x[i, ...]
    x_i = x_i[None, ...]
    
    with torch.no_grad():
        logit = model(x_i)
        logits.append(logit)

logits_batched = torch.cat(logits, dim=0)
print(logits_batched.shape)
with torch.no_grad():
    logits_orig = model(x)
print(torch.eq(logits_batched, logits_orig))
print(torch.allclose(logits_batched, logits_orig, atol=1e-5))

print(logits_batched[0, 0], logits_orig[0, 0])
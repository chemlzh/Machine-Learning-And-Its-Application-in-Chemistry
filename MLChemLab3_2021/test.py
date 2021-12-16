import os
import sys
import torchvision
from torch.utils.data import Dataset, DataLoader

os.chdir(sys.path[0])
workdir=".\\workdir"
if not os.path.exists(os.path.join(workdir, "__init__.py")):
    f = open(os.path.join(workdir, "__init__.py"), "w")
    f.close()

transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
testset = torchvision.datasets.MNIST('.\\data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size = 32, shuffle = False, drop_last = False)

# trainset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transform)
# trainloader = DataLoader(trainset, batch_size = 32, shuffle = True, drop_last = True)

from workdir.cnn_mnist import CNN

model = CNN()
# model.fit(trainloader, testloader, max_epoch=3, checkpoints_path = "workdir/checkpoints")
model.load_checkpoint(path = "workdir\\checkpoints\\model.pt")
acc, prec, rec, cm, mis, pred = model.evaluation(testloader)

print(acc)

myacc = model.eval_handwritten_digits("workdir\\handwritten")
print("Accuracy on custom dataset: ", myacc)
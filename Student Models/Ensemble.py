import Helper
from accs import net_st1
from gyrs import net_st2
from mags import net_st3



class Ensemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(input, 16)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3

        x = self.fc1(out)
        return torch.softmax(x, dim=1)
    

model = Ensemble(net_st1, net_st2 , net_st3, 16)


model.to(device)


optimizer = optim.Adam(model.parameters(),lr=0.003)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

model, train_loss, test_loss = Helper.train(model, train_loader, test_loader, epoch, optimizer, criterion)

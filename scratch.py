import torch


class AverageMeter:
    def __init__(self, decimals=3):
        self.value = torch.tensor(0).float()
        self.decimals = decimals
        self.n = 0

    def update(self, val):
        self.value += val
        self.n += 1

    def get_value(self):
        return round((self.value / self.n).item(), self.decimals)

    def reset(self):
        self.value = 0


test = torch.rand([1000])

print(test.mean())

avgmeter = AverageMeter()
for t in test:
    avgmeter.update(t)
print(avgmeter.get_value())

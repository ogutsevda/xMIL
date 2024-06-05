import torch


class BinaryClassifier:

    def __init__(self, model, learning_rate, weight_decay, objective='cross-entropy', gradient_clip=None,
                 device=torch.device('cpu')):
        self.model = model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        if objective == 'cross-entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        self.objective = objective
        self.gradient_clip = gradient_clip
        self.device = device
        self.model.to(self.device)

    def training_step(self, batch):
        raise NotImplementedError()

    def validation_step(self, batch, softmax=True):
        raise NotImplementedError()

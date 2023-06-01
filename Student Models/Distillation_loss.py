import torch
from torch import nn 
from torch.nn import functional as F

class DistillationLoss:
    
    def __init__(self):
        self.student_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss()
        self.temperature = 1
        self.alpha = 0.25

    def __call__(self, student_logits, student_target_loss, teacher_logits):
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))

        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return loss
    
distl = DistillationLoss()
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ArcfaceLossSimple(nn.Module):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0):
        
        super(ArcfaceLossSimple, self).__init__()
        self.margin = margin
        self.scale = scale
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.margin_cos = torch.cos(margin)
        self.margin_sin = torch.sin(margin)
        self.threshold = torch.cos(np.pi - margin)
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = torch.zeros([batch_size], dtype=torch.int64)

    def forward(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.copy_(torch.argmax(y_true, dim=-1))
        
        pick_cond = torch.where(y_true != 0)
       
        y_pred_vals = norm_logits[pick_cond]
        
        theta = y_pred_vals * self.margin_cos - torch.sqrt(1 - torch.pow(y_pred_vals, 2)) * self.margin_sin
        
        theta_valid = torch.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        
        arcface_logits = norm_logits.clone()
        
        arcface_logits[pick_cond] = theta_valid
        
        arcface_logits *= self.scale
        
        return F.cross_entropy(arcface_logits, y_true, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

class ArcfaceLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0):
        super(ArcfaceLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        self.scale = scale
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.threshold = torch.cos((torch.pi - margin2) / margin1)
        self.theta_min = (-1 - margin3) * 2

    def forward(self, y_true, norm_logits):
        
        batch_labels_back_up = None
        if self.training:
            
            batch_labels_back_up = y_true.argmax(dim=-1)
        
        pick_cond = torch.where(y_true != 0)
        
        y_pred_vals = torch.gather(norm_logits, pick_cond)
        
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = y_pred_vals
       
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = torch.cos(torch.acos(y_pred_vals) + self.margin2)
        
        else:
            theta = torch.cos(torch.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3
        
        theta_valid = torch.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        
        arcface_logits = norm_logits.clone()
       
        arcface_logits[pick_cond] = theta_valid
        
        arcface_logits *= self.scale
        
        loss = F.cross_entropy(arcface_logits, y_true, reduction='mean', label_smoothing=self.label_smoothing)
        
        return loss

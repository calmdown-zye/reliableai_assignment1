


import torch
import torch.nn as nn

def fgsm_targeted(model, x, target, eps):

    model.eval()
    
    x_adv = x.clone().requires_grad_(True)
    
    # 1. forward pass → loss 계산 (target 기준)
    
    output = model(x_adv)
    
    criterion = nn.CrossEntropyLoss() # p(target|x_adv) 최대화 → loss 최소화 (loss = -log(p(target|x_adv)))
    loss = criterion(output, target)
    
    # 2. backward 
    loss.backward()
    
    # 3. x_adv = x - eps * sign(x.grad) and clamp
    x_adv = x_adv - eps * x_adv.grad.sign()
    x_adv = x_adv.clamp(0,1)
    
    return x_adv.detach()
    
 

def fgsm_untargeted(model, x, label, eps):

    model.eval()
    
    x_adv = x.clone().requires_grad_(True)
    
    # 1. forward pass → loss 계산 (target 기준)
    
    output = model(x_adv)
    
    criterion = nn.CrossEntropyLoss() # p(target|x_adv) 최대화 → loss 최소화 (loss = -log(p(target|x_adv)))
    loss = criterion(output, label)
    
    # 2. backward 
    loss.backward()
    
    # 3. x_adv = x - eps * sign(x.grad) and clamp
    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = x_adv.clamp(0,1)
    
    return x_adv.detach()
import torch
import torch.nn as nn






def pgd_targeted(model, x, target, k, eps, eps_step):
    model.eval()
    
    # 시작점: 원본 이미지
    x_adv = x.clone()
    
    for i in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        
        # 1. forward pass
        output = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # 2. backward
        loss.backward()
        
        # 3. FGSM step 
        x_adv = x_adv - eps_step * x_adv.grad.sign()
        
        # 4. ε-ball projection (원본 기준으로 클리핑)
        #    x_adv가 원본 x의 ε 범위를 벗어나지 않도록
        eta = x_adv - x
        eta = eta.clamp(-eps,eps)
        x_adv = x + eta
        
        # 5. 유효 픽셀 범위
        x_adv = x_adv.clamp(0, 1)
    
    return x_adv.detach()


def pgd_untargeted(model, x, label, k, eps, eps_step):
    model.eval()
    
    # 시작점: 원본 이미지
    x_adv = x.clone()
    
    for i in range(k):
        x_adv = x_adv.detach().requires_grad_(True)
        
        # 1. forward pass
        output = model(x_adv)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        
        # 2. backward
        loss.backward()
        
        # 3. FGSM step 
        x_adv = x_adv + eps_step * x_adv.grad.sign()
        
        # 4. ε-ball projection (원본 기준으로 클리핑)
        #    x_adv가 원본 x의 ε 범위를 벗어나지 않도록
        eta = x_adv - x
        eta = eta.clamp(-eps,eps)
        x_adv = x + eta
        
        # 5. 유효 픽셀 범위
        x_adv = x_adv.clamp(0, 1)
    
    return x_adv.detach()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from models.mnist_model import MNISTClassifier
from torchvision import datasets, transforms
from models.cifar_model import get_cifar_model

from attacks.fgsm import fgsm_targeted, fgsm_untargeted
from attacks.pgd import pgd_targeted, pgd_untargeted



# ── 디바이스 설정 ──────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() 
                       else "mps" if torch.backends.mps.is_available() 
                       else "cpu")
print(f"Using device: {device}")


## MNIST 학습
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor()            
    ])
    train_set = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False,
                                download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_mnist(model, train_loader, epochs=5):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {total_loss/len(train_loader):.4f}")




## CIFAR-10 학습

def get_cifar_loaders(batch_size=64):
    # CIFAR-10 전처리 (ResNet 입력 기준)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # 데이터 증강
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_cifar(model, train_loader, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}]  "
              f"Loss: {total_loss/len(train_loader):.4f}")

# ── 평가 함수 ───────────────────────────────────────────
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# ── Attack success rate 측정 ────────────────────────────
def evaluate_attack(model, test_loader, attack_fn, attack_kwargs,
                    targeted=False, target_class=None, n_samples=100):
    """
    attack_fn    : 공격 함수
    attack_kwargs: 공격 함수에 넘길 인자 (eps 등)
    targeted     : targeted 공격 여부
    n_samples    : 평가할 샘플 수
    """
    model.eval()
    success = 0
    total = 0

    for images, labels in test_loader:
        if total >= n_samples:
            break

        images, labels = images.to(device), labels.to(device)

        for i in range(len(images)):
            if total >= n_samples:
                break

            x = images[i:i+1]   # (1, C, H, W)
            label = labels[i:i+1]

            # targeted: 정답이 아닌 target class 설정
            if targeted:
                target = torch.tensor(
                    [(label.item() + 1) % 10]
                ).to(device)
                x_adv = attack_fn(model, x, target, **attack_kwargs)
                pred = model(x_adv).argmax(dim=1)
                if pred.item() == target.item():
                    success += 1
            else:
                x_adv = attack_fn(model, x, label, **attack_kwargs)
                pred = model(x_adv).argmax(dim=1)
                if pred.item() != label.item():
                    success += 1

            total += 1

    rate = 100 * success / total
    return rate

# ── 시각화 저장 ─────────────────────────────────────────
def save_visualizations(model, test_loader, attack_fn, attack_kwargs,
                        targeted, save_dir, prefix, n=5):
    """
    원본 / adversarial / perturbation 을 나란히 저장
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    for images, labels in test_loader:
        if count >= n:
            break
        images, labels = images.to(device), labels.to(device)

        for i in range(len(images)):
            if count >= n:
                break

            x = images[i:i+1]
            label = labels[i:i+1]

            if targeted:
                target = torch.tensor(
                    [(label.item() + 1) % 10]
                ).to(device)
                x_adv = attack_fn(model, x, target, **attack_kwargs)
            else:
                x_adv = attack_fn(model, x, label, **attack_kwargs)

            # 예측값
            pred_orig = model(x).argmax(dim=1).item()
            pred_adv  = model(x_adv).argmax(dim=1).item()

            # perturbation 시각화 (차이를 10배 확대)
            perturb = (x_adv - x).abs() * 10

            # CPU로 이동 후 numpy 변환
            def to_img(t):
                t = t.squeeze().cpu().detach()
                if t.dim() == 3:       # CIFAR: (3, H, W)
                    return t.permute(1, 2, 0).numpy().clip(0, 1)
                else:                  # MNIST: (H, W)
                    return t.numpy().clip(0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(to_img(x),       cmap='gray' if x.shape[1]==1 else None)
            axes[0].set_title(f"Original\npred: {pred_orig}")
            axes[1].imshow(to_img(x_adv),   cmap='gray' if x.shape[1]==1 else None)
            axes[1].set_title(f"Adversarial\npred: {pred_adv}")
            axes[2].imshow(to_img(perturb),  cmap='gray' if x.shape[1]==1 else None)
            axes[2].set_title("Perturbation (×10)")

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            path = os.path.join(save_dir, f"{prefix}_sample{count}.png")
            plt.savefig(path)
            plt.close()
            count += 1

    print(f"  시각화 저장 완료: {save_dir}/{prefix}_sample0~{n-1}.png")


# ── 메인 ────────────────────────────────────────────────
if __name__ == "__main__":

    # ── MNIST ──────────────────────────────────────────
    print("\n=== MNIST ===")
    mnist_model = MNISTClassifier().to(device)

    if os.path.exists("mnist_model.pth"):
        mnist_model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        print("저장된 모델 로드 완료")
    else:
        train_loader, test_loader = get_mnist_loaders()
        print("학습 시작...")
        train_mnist(mnist_model, train_loader, epochs=5)
        acc = evaluate(mnist_model, test_loader)
        if acc >= 95:
            torch.save(mnist_model.state_dict(), "mnist_model.pth")
            print(f"모델 저장 완료: mnist_model.pth ({acc:.2f}%)")
        else:
            print(f"정확도 부족 ({acc:.2f}%). 학습 재시도 필요.")

    train_loader, test_loader = get_mnist_loaders()
    evaluate(mnist_model, test_loader)

    # ── CIFAR-10 ────────────────────────────────────────
    print("\n=== CIFAR-10 ===")
    cifar_model = get_cifar_model(device)
    cifar_train_loader, cifar_test_loader = get_cifar_loaders()

    if os.path.exists("cifar_model.pth"):
        cifar_model.load_state_dict(torch.load("cifar_model.pth", map_location=device))
        print("저장된 모델 로드 완료")
    else:
        print("학습 시작...")
        train_cifar(cifar_model, cifar_train_loader, epochs=10)
        cifar_acc = evaluate(cifar_model, cifar_test_loader)
        if cifar_acc >= 80:
            torch.save(cifar_model.state_dict(), "cifar_model.pth")
            print(f"모델 저장 완료: cifar_model.pth ({cifar_acc:.2f}%)")
        else:
            print(f"정확도 부족 ({cifar_acc:.2f}%). 학습 재시도 필요.")

    evaluate(cifar_model, cifar_test_loader)

    # ── 공격 평가 ──────────────────────────────────────
    print("\n=== 공격 평가 시작 ===")

    attacks_config = [
        ("FGSM_targeted",   fgsm_targeted,   {"eps": 0.1}, True),
        ("FGSM_untargeted", fgsm_untargeted, {"eps": 0.1}, False),
        ("PGD_targeted",    pgd_targeted,
         {"k": 40, "eps": 0.3, "eps_step": 0.01}, True),
        ("PGD_untargeted",  pgd_untargeted,
         {"k": 40, "eps": 0.3, "eps_step": 0.01}, False),
    ]

    for name, fn, kwargs, targeted in attacks_config:
        print(f"\n[MNIST] {name}")
        rate = evaluate_attack(mnist_model, test_loader,
                               fn, kwargs, targeted, n_samples=100)
        print(f"  Success rate: {rate:.1f}%")
        save_visualizations(mnist_model, test_loader,
                            fn, kwargs, targeted,
                            save_dir="results",
                            prefix=f"mnist_{name}")

        print(f"[CIFAR-10] {name}")
        rate = evaluate_attack(cifar_model, cifar_test_loader,
                               fn, kwargs, targeted, n_samples=100)
        print(f"  Success rate: {rate:.1f}%")
        save_visualizations(cifar_model, cifar_test_loader,
                            fn, kwargs, targeted,
                            save_dir="results",
                            prefix=f"cifar_{name}")

    # ── ε 테이블 실험 (report용) ─────────────────────────
    print("\n=== ε 테이블 실험 ===")
    eps_values = [0.05, 0.1, 0.2, 0.3]

    print(f"\n{'eps':<6} {'MNIST_FGSM_T':>12} {'MNIST_FGSM_U':>12} {'MNIST_PGD_T':>11} {'MNIST_PGD_U':>11} {'CIFAR_FGSM_T':>12} {'CIFAR_FGSM_U':>12} {'CIFAR_PGD_T':>11} {'CIFAR_PGD_U':>11}")
    print("-" * 105)

    for eps in eps_values:
        m_fgsm_t = evaluate_attack(mnist_model, test_loader, fgsm_targeted,
                                    {"eps": eps}, targeted=True, n_samples=100)
        m_fgsm_u = evaluate_attack(mnist_model, test_loader, fgsm_untargeted,
                                    {"eps": eps}, targeted=False, n_samples=100)
        m_pgd_t  = evaluate_attack(mnist_model, test_loader, pgd_targeted,
                                    {"k": 40, "eps": eps, "eps_step": eps/10},
                                    targeted=True, n_samples=100)
        m_pgd_u  = evaluate_attack(mnist_model, test_loader, pgd_untargeted,
                                    {"k": 40, "eps": eps, "eps_step": eps/10},
                                    targeted=False, n_samples=100)
        c_fgsm_t = evaluate_attack(cifar_model, cifar_test_loader, fgsm_targeted,
                                    {"eps": eps}, targeted=True, n_samples=100)
        c_fgsm_u = evaluate_attack(cifar_model, cifar_test_loader, fgsm_untargeted,
                                    {"eps": eps}, targeted=False, n_samples=100)
        c_pgd_t  = evaluate_attack(cifar_model, cifar_test_loader, pgd_targeted,
                                    {"k": 40, "eps": eps, "eps_step": eps/10},
                                    targeted=True, n_samples=100)
        c_pgd_u  = evaluate_attack(cifar_model, cifar_test_loader, pgd_untargeted,
                                    {"k": 40, "eps": eps, "eps_step": eps/10},
                                    targeted=False, n_samples=100)

        print(f"{eps:<6} {m_fgsm_t:>11.1f}% {m_fgsm_u:>11.1f}% {m_pgd_t:>10.1f}% {m_pgd_u:>10.1f}% {c_fgsm_t:>11.1f}% {c_fgsm_u:>11.1f}% {c_pgd_t:>10.1f}% {c_pgd_u:>10.1f}%", flush=True)
        
        
        
        # ── ε별 시각화 저장 (CIFAR FGSM targeted) ──────────────
        print("\n=== ε별 시각화 저장 ===")
        for eps in [0.05, 0.1, 0.2, 0.3]:
            save_visualizations(
                cifar_model, cifar_test_loader,
                fgsm_targeted, {"eps": eps},
                targeted=True,
                save_dir="results",
                prefix=f"cifar_FGSM_T_eps{eps}",
                n=5
            )
            print(f"ε={eps} 저장 완료", flush=True)
            
            


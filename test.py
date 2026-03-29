import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mnist_model import MNISTClassifier

# ── 디바이스 설정 ──────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── MNIST 데이터 로드 ───────────────────────────────────
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),              # [0,255] → [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균/표준편차
    ])
    train_set = datasets.MNIST(root='./data', train=True,
                                download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False,
                                download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ── 학습 함수 ───────────────────────────────────────────
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

# ── 메인 ────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()

    # MNIST 학습
    print("\n=== MNIST 학습 시작 ===")
    mnist_model = MNISTClassifier().to(device)
    train_mnist(mnist_model, train_loader, epochs=5)
    acc = evaluate(mnist_model, test_loader)

    # 모델 저장 (나중에 공격 때 재사용)
    if acc >= 95:
        torch.save(mnist_model.state_dict(), "mnist_model.pth")
        print("모델 저장 완료: mnist_model.pth")
    else:
        print(f"정확도 부족 ({acc:.2f}%). 학습 재시도 필요.")
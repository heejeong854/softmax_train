import streamlit as st
import torch
from torch import nn, optim
from torchvision import datasets, transforms

# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 64),  # 은닉층 크기 줄임
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 데이터 준비 (일부만)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
subset = torch.utils.data.Subset(train_dataset, range(1000))  # 1000장만 사용
train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)  # 배치 128

# 모델, 손실함수, 옵티마이저
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

st.title("빠른 MNIST 숫자 인식 학습")

if st.button("1 epoch 빠른 학습 시작"):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    st.success("1 epoch 학습 완료!")

# 테스트 이미지 예측
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_img, test_label = test_dataset[0]
st.image(test_img.squeeze().numpy(), caption=f"정답: {test_label}")

model.eval()
with torch.no_grad():
    logits = model(test_img.unsqueeze(0))
    pred = torch.argmax(logits, dim=1).item()
st.write(f"모델 예측: {pred}")


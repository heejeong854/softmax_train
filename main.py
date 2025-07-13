import streamlit as st
import torch
from torch import nn, optim
from torchvision import datasets, transforms

# 모델 정의 (간단한 MLP)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 데이터셋 준비
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Streamlit UI
st.title("간단한 MNIST 숫자 분류기")

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

if st.button("학습 시작 (1 epoch)"):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    st.success("1 epoch 학습 완료!")

# 학습 후 테스트 이미지 하나 예측
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_img, test_label = test_dataset[0]
st.image(test_img.squeeze().numpy(), caption=f"정답: {test_label}")

model.eval()
with torch.no_grad():
    logits = model(test_img.unsqueeze(0))
    pred = torch.argmax(logits, dim=1).item()
st.write(f"모델 예측: {pred}")

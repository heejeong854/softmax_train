import streamlit as st
import torch
from torch import nn, optim
from torchvision import datasets, transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),  # 은닉층 128
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

st.title("MNIST 숫자 인식 - 더 크게 학습")

if st.button("5 epoch 학습 시작"):
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
    st.success("5 epoch 학습 완료!")

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_img, test_label = test_dataset[0]
st.image(test_img.squeeze().numpy(), caption=f"정답: {test_label}")

model.eval()
with torch.no_grad():
    logits = model(test_img.unsqueeze(0))
    pred = torch.argmax(logits, dim=1).item()
st.write(f"모델 예측: {pred}")

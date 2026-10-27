# Deep learning and neural networks

## A neuron and a network
A neuron computes y = σ(w·x + b). Stacking layers with non-linear activations (ReLU = max(0, z), sigmoid, tanh, softmax) lets a network approximate almost any function.

## Training
1. **Forward pass:** compute the prediction and the loss (MSE for regression, cross-entropy for classification).
2. **Backpropagation:** the chain rule gives ∂loss/∂w for every weight.
3. **Gradient descent:** w ← w − η ∂L/∂w, in mini-batches (SGD), usually with **Adam** (momentum plus a per-parameter learning rate).
4. Repeat for several epochs. Watch the validation loss to catch overfitting.

```python
model = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb, yb in loader:
    loss = nn.functional.mse_loss(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
```

## Making it work
- **Scale the inputs,** and initialise weights sensibly.
- **Regularise:** dropout, weight decay, early stopping, data augmentation.
- **Batch normalisation** for faster, more stable training. **Gradient clipping** for exploding gradients.
- **Architectures:** CNNs for images (convolution + pooling), RNN / LSTM for sequences, and Transformers (self-attention) for language, the basis of LLMs.

**Applied elsewhere:** the [FinSearch project](https://github.com/ManasJ-2007/finsearch-rl-trading) uses an LSTM forecaster, and [summer-of-science-rl](https://github.com/ManasJ-2007/summer-of-science-rl) trains PyTorch networks for DQN, A2C, PPO and DDPG.

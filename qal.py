
import torch
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# Toy Dataset: XOR-like classification
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)  # Binary labels

# Quantum State Initialization
n_qubits = 2  # Small number of qubits
hilbert_dim = 2 ** n_qubits
psi = F.normalize(torch.randn(hilbert_dim, dtype=torch.cfloat), dim=0)  # Random initial state

# Encoding Function: Convert classical data to unitary evolution
# Here, we use a simple diagonal Hamiltonian encoding.
def encode_data(x):
    H_x = torch.diag(torch.tensor([x[0], x[1], -x[0], -x[1]], dtype=torch.float32))
    return torch.linalg.matrix_exp(-1j * H_x * torch.pi / 4)  # Simulating quantum evolution

# Target-Oriented Perturbation
def target_perturbation(psi, y, eta):
    P_y = torch.zeros_like(psi)
    P_y[y] = 1  # Projection onto the correct label
    return F.normalize((1 - eta) * psi + eta * P_y, dim=0)

# Training Parameters
eta = 0.1  # Learning rate
num_epochs = 50

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(X)):
        x, y = X[i], int(Y[i].item())
        U_x = encode_data(x)
        psi_evolved = U_x @ psi  # Quantum state evolution
        
        # Measurement (classification): Probability of measuring |1>
        prob = torch.abs(psi_evolved[1])**2 + torch.abs(psi_evolved[3])**2
        loss = (prob - y) ** 2  # Squared error loss
        total_loss += loss.item()
        
        # Target-Oriented Perturbation Step
        psi = target_perturbation(psi, y, eta)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# Final Inference
def predict(x):
    U_x = encode_data(x)
    psi_evolved = U_x @ psi
    prob = torch.abs(psi_evolved[1])**2 + torch.abs(psi_evolved[3])**2
    return 1 if prob > 0.5 else 0

print("Predictions:")
for i in range(len(X)):
    print(f"Input {X[i].tolist()} -> Predicted: {predict(X[i])}, Actual: {Y[i].item()}")

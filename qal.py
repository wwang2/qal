import torch
import torch.nn.functional as F

# Set random seed for reproducibility
# torch.manual_seed(42)

# Generate a toy dataset: 10 samples, each with 2 features.
num_samples = 20
X = torch.rand((num_samples, 2), dtype=torch.float32)
Y = (X[:, 0] > 0.5).float()  # Label: 1 if first feature > 0.5, else 0

# Increase the number of qubits.
n_qubits = 4  # For example, change this to any number.
hilbert_dim = 2 ** n_qubits  # Dimension of the Hilbert space.

# Initialize a random normalized quantum state (of dimension hilbert_dim).
psi = F.normalize(torch.randn(hilbert_dim, dtype=torch.cfloat), dim=0)

def encode_data(x, n_qubits):
    """
    Create a unitary operator U from a 2-dimensional sample x.
    For a larger Hilbert space, we tile a base diagonal pattern.
    """
    hilbert_dim = 2 ** n_qubits
    # Define a base pattern from the two features.
    base_diag = torch.tensor([x[0], x[1], -x[0], -x[1]], dtype=torch.float32)
    repeats = hilbert_dim // base_diag.shape[0]
    diag_elements = base_diag.repeat(repeats)
    # If hilbert_dim is not a multiple of the base pattern's length, pad with zeros.
    if diag_elements.shape[0] < hilbert_dim:
        pad_size = hilbert_dim - diag_elements.shape[0]
        diag_elements = torch.cat([diag_elements, torch.zeros(pad_size, dtype=torch.float32)])
    
    # Create a diagonal Hamiltonian of size (hilbert_dim x hilbert_dim).
    H_x = torch.diag(diag_elements.to(torch.cfloat))
    
    # Generate the unitary operator via quantum evolution:
    # U = exp(-i * H_x * (π/4))
    U = torch.linalg.matrix_exp(-1j * H_x * torch.pi / 4)
    return U

def measure_probability(psi_evolved, n_qubits):
    """
    Compute the probability of measuring label 1.
    Here, we assume that label 1 corresponds to the measurement outcomes 
    where the last qubit is 1 (i.e. basis indices that are odd numbers).
    """
    prob = 0.0
    for i in range(len(psi_evolved)):
        if i % 2 == 1:  # If the last bit is 1 (i.e. index is odd).
            prob += torch.abs(psi_evolved[i])**2
    return prob

def target_perturbation(psi, y, eta, n_qubits):
    """
    Update the quantum state psi by nudging it toward the subspace corresponding
    to the true label y. If y == 1, we target all basis states with an odd index;
    if y == 0, we target those with an even index.
    """
    P_y = torch.zeros_like(psi)
    for i in range(len(psi)):
        if i % 2 == y:  # Even indices for label 0; odd indices for label 1.
            P_y[i] = 1.0
    # Normalize the projection to form a valid quantum state.
    P_y = F.normalize(P_y, dim=0)
    
    # Mix the current state and the target state.
    psi_new = (1 - eta) * psi + eta * P_y
    # Renormalize the updated state.
    return F.normalize(psi_new, dim=0)

# Training parameters.
eta = 0.1  # Learning rate for the state update.
num_epochs = 200

# Training loop.
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(num_samples):
        x, y = X[i], int(Y[i].item())
        # Get the unitary evolution operator for the sample.
        U = encode_data(x, n_qubits)
        # Evolve the state.
        psi_evolved = U @ psi
        # Measure the probability for label 1.
        prob = measure_probability(psi_evolved, n_qubits)
        # Compute a simple squared error loss.
        loss = (prob - y) ** 2  
        total_loss += loss.item()
        
        # Update psi by steering it toward the target measurement outcome.
        psi = target_perturbation(psi_evolved, y, eta, n_qubits)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

# Inference function: use the final psi to predict labels.
def predict(x, n_qubits):
    U = encode_data(x, n_qubits)
    psi_evolved = U @ psi
    prob = measure_probability(psi_evolved, n_qubits)
    return 1 if prob > 0.5 else 0

print("\nPredictions:")
for i in range(num_samples):
    print(f"Input {X[i].tolist()} -> Predicted: {predict(X[i], n_qubits)}, Actual: {int(Y[i].item())}")

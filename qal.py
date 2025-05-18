import torch
import torch.nn.functional as F

# Set random seed for reproducibility
# torch.manual_seed(42)

# Generate a toy dataset: 10 samples, each with 2 features.
num_samples = 10
X = torch.randint(0, 2, (num_samples, 2), dtype=torch.float32)
Y = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)  # Label: 1 if first feature > 0.5, else 0

# Increase the number of qubits.
n_qubits = 2  # For example, change this to any number.
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

def target_perturbation(psi_evolved, y, eta, n_qubits):
    """
    Perform the target-oriented perturbation on the evolved state psi_evolved.
    Projects psi_evolved onto the subspace where the last qubit equals y,
    then mixes with the original evolved state according to learning rate eta.
    """
    # Project evolved state onto subspace for label y (even indices for y=0, odd for y=1)
    proj = torch.zeros_like(psi_evolved)
    proj[y::2] = psi_evolved[y::2]
    # Mix original and projected states
    psi_temp = (1 - eta) * psi_evolved + eta * proj
    return psi_temp

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

        # Perform target perturbation directly using psi_evolved
        # Project evolved state onto label subspace and mix
        proj = torch.zeros_like(psi_evolved)
        proj[y::2] = psi_evolved[y::2]
        psi_temp = (1 - eta) * psi_evolved + eta * proj
        # Map back to original basis before next sample
        psi = U.conj().T @ psi_temp
        psi = F.normalize(psi, dim=0)
    
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

Here’s a concise, equation-based walkthrough of the QAL training loop, matching the steps in your README to the paper’s formalism:

⸻

1. Data‐Dependent Hamiltonian and Loss

For each sample (x,y), define the data‐encoded Hamiltonian
H_x \;=\; I \;-\; U(x)^\dagger\,\Pi_y\,U(x),
where \Pi_y projects onto the \log k-qubit subspace corresponding to label y.
The per‐sample failure probability (loss) is
\ell_x(ψ) \;=\; 1 - \langle ψ|\,U(x)^\dagger\,\Pi_y\,U(x)\,|ψ\rangle
\;=\;\langle ψ|\,H_x\,|ψ\rangle.
Averaging over the training set S, the empirical risk is
\[
\widehat R_S(ψ)
\;=\;\frac1N\sum_{(x,y)\in S}\ell_x(ψ)
\;=\;\bigl\langle ψ \bigm| H_S \bigm| ψ\bigr\rangle,
\quad
H_S \;=\;\mathbb{E}_{(x,y)\sim S}[\,H_x\,].
\]

⸻

2. Single‐Step Update as “Imaginary‐Time” Evolution

Starting from \lvert ψ\rangle, one training step with sample (x,y) applies:
	1.	Evolve under the data unitary:
\lvert ψ_e\rangle = U(x)\,\lvert ψ\rangle.
	2.	Non‐unitary perturbation toward the y-subspace:
\lvert ψ{\prime}_e\rangle
= (I - η\,H_x)\,\lvert ψ_e\rangle
\;=\;(1-η)\,\lvert ψ_e\rangle \;+\; η\,U(x)^\dagger\,\Pi_y\,U(x)\,\lvert ψ_e\rangle.
	3.	Inverse evolution back to the original basis and normalize:
\lvert ψ_{\rm new}\rangle
= \frac{U(x)^\dagger\,\lvert ψ{\prime}_e\rangle}{\bigl\|U(x)^\dagger\,\lvert ψ{\prime}_e\rangle\bigr\|}
= \frac{(I - η\,H_x)\,\lvert ψ\rangle}{\|(I - η\,H_x)\,\lvert ψ\rangle\|}.
Equation (1) in the paper summarizes this in one shot:
\[
\lvertψ\rangle \;\longmapsto\;
\frac{(I - η\,H_x)\,\lvertψ\rangle}{\|(I - η\,H_x)\,\lvertψ\rangle\|}.
\tag{1}
\]2502.05264v1 (1).pdf](file-service://file-6ohhsjU9eWEAr5WjqVEpUE)

⸻

3. Stochastic and Imaginary‐Time Perspective
	•	Density‐matrix form (unnormalized):
\[
ρ \;\longmapsto\; (I - η\,H_x)\,ρ\,(I - η\,H_x).
\]
	•	Averaging over random samples yields, to first order in η,
\[
ρ \;\longmapsto\; e^{-ηH_S}\,ρ\,e^{-ηH_S}
\quad\Longrightarrow\quad
ρ_T \;=\; e^{-ηT\,H_S}\,ρ_0\,e^{-ηT\,H_S},
\tag{2}
\]
which is exactly imaginary‐time evolution under H_S, guaranteeing exponential convergence to its ground state 2502.05264v1 (1).pdf](file-service://file-6ohhsjU9eWEAr5WjqVEpUE).

⸻

4. Measurement & Prediction

After training to obtain \lvertψ^*\rangle, inference on a new x{\prime} proceeds by:
	1.	Encode and evolve: \lvertψ_e\rangle = U(x{\prime})\,\lvertψ^*\rangle.
	2.	Measure the projector \Pi_{y{\prime}} on the last \log k qubits.
	3.	The predicted label is
\hat y = \arg\max_{y{\prime}} \langle ψ_e|\Pi_{y{\prime}}|ψ_e\rangle.

⸻

Key takeaways:
	•	The update (I - ηH_x) is a linear, gradient-free map that “cools” the state toward correct‐label subspaces.
	•	Repeated random sampling → stochastic imaginary‐time evolution → provable, exponential convergence to the ground state of H_S.
	•	Generalization error is tightly bounded as
\displaystyle \max_ψ\bigl[R(ψ)-\widehat R_S(ψ)\bigr]\le\sqrt{\frac{4\ln(2n+1/δ)}{N}}
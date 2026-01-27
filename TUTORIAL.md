# Tutorial: Understanding the Ψ-Field Simulation

## Introduction

This tutorial walks you through the simulation step-by-step, explaining the physics and code.

## Part 1: Quantum Basics

### The Double-Slit Experiment

```
   Particle Source    Double Slit       Screen
        ●    ━━━━━━━━━━━━━━━━━━━━━  ▓▓▓▓▓▓▓
        │               █             ▓▓▓▓▓▓▓
        │               ■ ← Left      ▓▓▓▓▓▓▓
        └────────→      █             ▓▓▓▓▓▓▓
                        █             ▓▓▓▓▓▓▓
                        ■ ← Right     ▓▓▓▓▓▓▓
                        █             ▓▓▓▓▓▓▓
                   Interference pattern
```

**Key Facts:**
- Particles behave like waves → interference pattern
- Measuring which slit → pattern disappears
- This is **Bohr's Complementarity Principle**

### Bohr's Complementarity

**Statement:** You cannot have BOTH:
1. Wave behavior (interference, V = 1)
2. Particle behavior (which-path info, I = 1)

**Mathematical form:**
```
V + I ≤ 1
```

where:
- **V** = Visibility of interference fringes
- **I** = Which-path information

## Part 2: The Ψ-Field Hypothesis

### The "Crazy" Idea

**Question:** What if we could measure the path WITHOUT destroying interference?

**Postulate:** Introduce a new observable χ that:
```python
[χ, x] = 0  # Commutes with position
[χ, p] = 0  # Commutes with momentum
```

**Implication:** Measuring χ wouldn't disturb x or p → no decoherence!

### The Paradox

**Problem:** If χ doesn't interact with position, how can it tell us which slit?

**Answer:** We don't know! This is the **thought experiment** part.

We postulate it works and explore the consequences.

## Part 3: Running the Simulation

### Step 1: Install

```bash
git clone https://github.com/yourusername/psi-field-experiment.git
cd psi-field-experiment
pip install -r requirements.txt
```

### Step 2: Run

```bash
python psi_field_simulator.py
```

### Step 3: Understand the Output

The simulation runs **three modes:**

#### Mode 1: Control
```
No measurement → Maximum interference
Expected: V ≈ 1, I = 0
```

#### Mode 2: Standard QM
```
Measure path WITH collapse → Reduced interference
Expected: V < 1, I > 0, but V + I ≤ 1
```

#### Mode 3: Ψ-Field
```
Measure path WITHOUT collapse → ???
Expected: V ≈ 1, I > 0, maybe V + I > 1?
```

### Step 4: Interpret Results

Look at the output:

```
Mode                    Visibility V     Path Info I           V + I
------------------------------------------------------------------------
Control                       0.9817          0.0000          0.9817
Standard QM                   0.9688          0.0115          0.9802
Ψ-Field                       0.9833          0.0294          1.0127
```

**Key observation:** Ψ-Field mode has V + I = 1.0127 > 1 ⚠️

## Part 4: Understanding the Code

### Main Components

```python
# 1. Quantum wavefunction evolution
class QuantumWaveFunction:
    def propagate_free_space(self, dt):
        # Uses FFT for solving Schrödinger equation
        
    def apply_double_slit(self, dt):
        # Split-operator method for potential
```

```python
# 2. The χ-detector (the "magic" part)
class ChiDetector:
    def detect_which_way(self, psi):
        # Samples from |ψ|² to determine path
        # WITHOUT collapsing the wavefunction
```

```python
# 3. The experiment
class PsiFieldExperiment:
    def run_single_particle(self, mode):
        # Propagates particle through slits
        # Applies (or doesn't apply) χ-measurement
        # Measures final position
```

### Key Variables

- `psi`: Complex wavefunction ψ(x)
- `V`: Potential barrier (slits)
- `chi_measured`: Which-way detector result (0=left, 1=right)
- `x_final`: Final position on screen

## Part 5: Analyzing Results

### Calculating Visibility

```python
def calculate_visibility(x_data):
    # Build histogram of positions
    hist, _ = np.histogram(x_data, bins=50)
    
    # Find max and min intensities
    I_max = np.max(hist)
    I_min = np.min(hist)
    
    # Visibility formula
    V = (I_max - I_min) / (I_max + I_min)
    return V
```

**Interpretation:**
- V ≈ 1: Clear fringes (wave behavior)
- V ≈ 0: No fringes (particle behavior)

### Calculating Which-Way Info

```python
def calculate_which_way_info(x_data, chi_data):
    # Split by position
    x_left = (x_data < median)  # Left of center?
    
    # Split by χ measurement
    chi_left = (chi_data == 0)  # Detected at left slit?
    
    # Correlation
    I = |correlation(x_left, chi_left)|
    return I
```

**Interpretation:**
- I ≈ 1: χ perfectly predicts position (particle behavior)
- I ≈ 0: χ tells nothing (wave behavior)

## Part 6: Customization

### Changing Parameters

Edit the `Config` class:

```python
@dataclass
class Config:
    # Try wider slits:
    slit_separation: float = 20.0  # Default: 12.0
    
    # Try more accurate detector:
    chi_fidelity: float = 0.99     # Default: 0.98
    
    # Try different momentum:
    k0: float = 30.0               # Default: 18.0
```

### Expected Effects

**Wider slits (larger `slit_separation`):**
- Easier to distinguish paths
- Higher I (more path info)
- Stronger violation if Ψ-field works

**Better detector (higher `chi_fidelity`):**
- More accurate which-way info
- Higher I
- Larger V + I excess

**Higher momentum (larger `k0`):**
- More directed beam
- Different fringe spacing
- May change visibility

## Part 7: Going Deeper

### Exercise 1: Verify Standard QM

Modify Mode 2 to collapse the wavefunction more strongly:

```python
# In run_single_particle(), standard_qm mode:
if chi_measured == 0:
    wf.psi[x > 0] = 0  # Kill ALL right-side amplitude
else:
    wf.psi[x < 0] = 0  # Kill ALL left-side amplitude
```

**Expected:** V should drop significantly, but V + I should still be ≤ 1.

### Exercise 2: Perfect Detector

Set `chi_fidelity = 1.0` (perfect detection).

**Question:** Does V + I increase further?

**Answer:** Yes! Perfect detector → higher I → larger violation.

### Exercise 3: No-Signaling Test

Create two entangled particles:

```python
# Initial state: |ψ⟩ = (|LL⟩ - |RR⟩)/√2
psi_A = ...  # Alice's particle
psi_B = ...  # Bob's particle
```

**Test:**
1. Alice measures χ_A (Ψ-field mode)
2. Check if Bob's position distribution changes

**Expected:** Should NOT change (no-signaling theorem)

**If it does:** Theory is inconsistent!

## Part 8: Common Questions

### Q1: Is this real physics?

**A:** No. This is a **thought experiment**. We postulate [χ,x]=0 and explore consequences. Real QM doesn't have such an observable.

### Q2: Why is the violation so small (1.27%)?

**A:** Even in the toy model, getting path info is hard! The detector has limited fidelity, particles spread, and correlation isn't perfect.

### Q3: Could we make V + I much bigger?

**A:** In principle, yes:
- Perfect detector (fidelity = 1.0)
- Larger slit separation
- Multiple measurements

But real QM would fight back: decoherence, environmental effects, etc.

### Q4: What about weak measurements?

**A:** Weak measurements are REAL and already known. They give partial path info with V + I ≤ 1. Our Ψ-field claims to break this bound, which is much stronger.

### Q5: Has anyone tried this experimentally?

**A:** Many experiments test complementarity. ALL support V + I ≤ 1 within errors. No hint of Ψ-field-like effects.

## Part 9: Further Reading

### Complementarity Principle
1. **Bohr (1928)** - "The Quantum Postulate and the Recent Development of Atomic Theory"
2. **Wootters & Zurek (1979)** - "Complementarity in the double-slit experiment"

### Weak Measurements
3. **Aharonov et al. (1988)** - "How the result of a measurement of a component of the spin of a spin-1/2 particle can turn out to be 100"
4. **Wiseman (2007)** - "Weak values, quantum trajectories, and the cavity-QED experiment"

### Experimental Tests
5. **Englert (1996)** - "Fringe Visibility and Which-Way Information: An Inequality"
6. **Dürr et al. (1998)** - "Origin of quantum-mechanical complementarity probed by a 'which-way' experiment"

### Advanced QM
7. **Nielsen & Chuang** - "Quantum Computation and Quantum Information"
8. **Peres** - "Quantum Theory: Concepts and Methods"

## Conclusion

This simulation lets you explore the boundary between wave and particle behavior. While the Ψ-field is hypothetical, understanding WHY it probably can't exist teaches you deep lessons about quantum mechanics.

**Key Takeaway:** Complementarity isn't just a rule — it's deeply woven into the fabric of quantum theory. Breaking it would require rewriting physics from the ground up.

Happy exploring! 🎓

---

*Questions? Open an issue on GitHub!*

# Contributing to Ψ-Field Thought Experiment

First off, thank you for considering contributing to this project! 

## Project Philosophy

This is an **educational thought experiment** exploring the boundaries of quantum mechanics. We're not claiming to have discovered new physics, but rather exploring "what if" scenarios to better understand complementarity.

## Ways to Contribute

### 1. Bug Reports

Found a bug? Great! Please include:
- Python version
- OS (Windows/Mac/Linux)
- Error message (full traceback)
- Steps to reproduce

**Example:**
```
**Bug:** Simulation crashes with large grid size

Python: 3.9.7
OS: Ubuntu 20.04
Error: MemoryError on line 234

Steps:
1. Set Config.N = 4096
2. Run simulation
3. Crash after Mode 2
```

### 2. Feature Requests

Have an idea? Open an issue with:
- **What:** Brief description
- **Why:** What problem does it solve?
- **How:** Rough implementation idea (optional)

**Example features:**
- Different detector models (SQUID, cavity QED)
- 3D visualization of wavefunction
- Parameter sweeps (automated testing)
- Export to real experimental parameters

### 3. Code Contributions

#### Before You Start

1. **Open an issue first** to discuss the feature
2. **Keep it focused** (one feature per PR)
3. **Follow the style** (see below)

#### Code Style

We follow PEP 8 with these additions:

```python
# Good: Descriptive names
def calculate_visibility(x_data: np.ndarray) -> float:
    """Calculate interference fringe visibility."""
    pass

# Bad: Unclear names
def calc_v(x):
    pass
```

**Key points:**
- Type hints for function arguments
- Docstrings for all public functions
- Comments explaining physics, not code
- NumPy-style docstrings

**Example:**
```python
def propagate_free_space(self, dt: float) -> None:
    """
    Evolve wavefunction in free space using FFT method.
    
    Applies kinetic energy operator in momentum space:
    ψ(k,t+dt) = exp(-i E(k) dt / ℏ) ψ(k,t)
    
    Parameters
    ----------
    dt : float
        Time step (atomic units)
        
    Notes
    -----
    Uses split-operator approximation. Error is O(dt³).
    """
    # Implementation...
```

#### Pull Request Process

1. **Fork** the repository
2. **Create branch:** `feature/your-feature-name`
3. **Write tests** (if applicable)
4. **Update docs** (README, docstrings)
5. **Submit PR** with clear description

**PR Template:**
```markdown
## Description
Brief summary of changes

## Motivation
Why is this needed?

## Changes
- Added X
- Modified Y
- Removed Z

## Testing
How did you test this?

## Checklist
- [ ] Code follows style guide
- [ ] Docstrings updated
- [ ] README updated (if needed)
- [ ] Tests pass
```

### 4. Documentation

Documentation improvements are highly valued!

**Areas that need help:**
- Clearer explanations of physics
- More examples in tutorial
- FAQs
- Non-English translations (we provide English source)

### 5. Educational Materials

Create supplementary content:
- Jupyter notebooks
- Video tutorials
- Blog posts explaining results
- Classroom exercises

**Share them!** We'll link to community resources.

## Testing

### Run Tests

```bash
# (when we have tests)
pytest tests/
```

### Manual Testing

Before submitting:
1. Run full simulation: `python psi_field_simulator.py`
2. Check plots look reasonable
3. Verify V + I values make sense
4. Test on different parameters

### What to Test

- **Physics:** Do results match expectations?
- **Numerics:** Is it stable? Convergent?
- **Edge cases:** What if N=1? k0=0? g=2.0?

## Physics Rigor

### Good Physics Practice

**DO:**
- Cite relevant papers
- Explain assumptions clearly
- Distinguish fact from speculation
- Show uncertainties

**DON'T:**
- Claim "proof" of new physics
- Overstate significance of results  
- Ignore known experimental constraints
- Make up mechanisms without justification

### Example

**Good:**
```python
# Postulate: χ commutes with x and p
# This violates standard QM measurement theory.
# We explore consequences IF it were true.
```

**Bad:**
```python
# Proof that QM is wrong!
# This simulation shows complementarity doesn't work!
```

## Discussion Guidelines

### Be Respectful

- Assume good faith
- Focus on ideas, not people
- Welcome beginners
- Celebrate curiosity

### Keep it Scientific

- Back claims with evidence/math
- Admit uncertainty
- Change mind based on arguments
- Separate "what is" from "what if"

### Examples

**Good discussion:**
> "Interesting idea! But how does χ avoid entanglement? In standard QM, 
> any measurement creates entanglement. Do you have a Hamiltonian that 
> avoids this?"

**Unhelpful:**
> "This is stupid, obviously wrong."

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone.

### Standards

**Positive:**
- Using welcoming language
- Respecting different viewpoints
- Accepting constructive criticism
- Showing empathy

**Unacceptable:**
- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Violations can be reported to project maintainers. All complaints will be reviewed and investigated.

## Questions?

Not sure about something? 

- **GitHub Issues:** For bugs, features, discussions
- **Email:** (if you provide one)
- **Discussions:** (if enabled on repo)

No question is too basic!

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in papers/presentations (if significant contribution)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Getting Started Checklist

Ready to contribute? Here's how:

- [ ] Read README.md and RESULTS.md
- [ ] Run the simulation successfully
- [ ] Read theory.md to understand physics
- [ ] Check existing issues/PRs
- [ ] Fork the repository
- [ ] Make your changes
- [ ] Test thoroughly
- [ ] Submit PR!

**Thank you for helping make quantum mechanics more accessible!** 🎓

---

*"The opposite of a correct statement is a false statement. But the opposite of a profound truth may well be another profound truth." — Niels Bohr*

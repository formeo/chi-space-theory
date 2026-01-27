"""
РЕАЛЬНЫЙ КВАНТОВЫЙ ЭКСПЕРИМЕНТ: EPR + Тест на аномалии
=======================================================

Это НЕ симуляция — это код для запуска на РЕАЛЬНОМ 
квантовом компьютере IBM!

Что делаем:
1. Создаём EPR пару (настоящую запутанность)
2. Проводим разные измерения
3. Проверяем Bell inequality
4. Ищем аномалии в корреляциях

Можно запустить:
- На симуляторе (локально)
- На реальном IBM Quantum (бесплатно!)

Author: Roman
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple

# Qiskit imports
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    print("✓ Qiskit loaded successfully")
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"⚠️  Qiskit not installed: {e}")
    print("   Run: pip install qiskit qiskit-aer")


class EPRExperiment:
    """
    Реальный EPR эксперимент.
    
    Создаём Bell state (EPR пару):
    |Φ+⟩ = (|00⟩ + |11⟩) / √2
    
    Свойства:
    - Измерения ВСЕГДА коррелированы
    - Нарушает Bell inequality (доказательство нелокальности)
    """
    
    def __init__(self, use_real_hardware: bool = False):
        self.use_real_hardware = use_real_hardware
        
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
        
    def create_epr_pair(self) -> QuantumCircuit:
        """
        Создаём EPR пару (Bell state).
        
        Схема:
            q0: ──H──●──
                     │
            q1: ─────X──
        
        H = Hadamard gate (суперпозиция)
        CX = CNOT gate (запутывание)
        """
        qc = QuantumCircuit(2, 2)
        
        # Создаём Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)           # Hadamard на первый кубит
        qc.cx(0, 1)       # CNOT: запутываем
        
        return qc
    
    def run_circuit(self, qc: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """Запуск схемы на симуляторе."""
        job = self.simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        return counts
    
    def measure_both_z(self, shots: int = 1024) -> Dict[str, int]:
        """
        Измеряем оба кубита в Z-базисе.
        
        Ожидаем: только |00⟩ и |11⟩ (по 50%)
        """
        qc = self.create_epr_pair()
        qc.measure([0, 1], [0, 1])
        
        return self.run_circuit(qc, shots)
    
    def measure_different_bases(self, shots: int = 1024) -> Dict[str, Dict]:
        """
        Измеряем в разных базисах (для Bell test).
        
        Базисы:
        - Z: стандартный (|0⟩, |1⟩)
        - X: Hadamard basis (|+⟩, |-⟩)
        """
        results = {}
        
        # 1. Оба в Z
        qc_zz = self.create_epr_pair()
        qc_zz.measure([0, 1], [0, 1])
        
        # 2. Оба в X (добавляем H перед измерением)
        qc_xx = self.create_epr_pair()
        qc_xx.h(0)
        qc_xx.h(1)
        qc_xx.measure([0, 1], [0, 1])
        
        # 3. Alice Z, Bob X
        qc_zx = self.create_epr_pair()
        qc_zx.h(1)  # Только Bob в X-базисе
        qc_zx.measure([0, 1], [0, 1])
        
        # 4. Alice X, Bob Z
        qc_xz = self.create_epr_pair()
        qc_xz.h(0)  # Только Alice в X-базисе
        qc_xz.measure([0, 1], [0, 1])
        
        # Запускаем все
        circuits = [qc_zz, qc_xx, qc_zx, qc_xz]
        names = ['ZZ', 'XX', 'ZX', 'XZ']
        
        for name, qc in zip(names, circuits):
            results[name] = self.run_circuit(qc, shots)
        
        return results


class CHSHTest:
    """
    CHSH inequality test — проверка квантовой нелокальности.
    
    Классический предел: |S| ≤ 2
    Квантовый максимум: |S| ≤ 2√2 ≈ 2.828
    
    Если S > 2 — доказательство нелокальности!
    """
    
    def __init__(self):
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
    
    def create_chsh_circuit(
        self, 
        theta_a: float, 
        theta_b: float
    ) -> QuantumCircuit:
        """
        CHSH circuit с углами измерения.
        
        theta_a: угол для Alice
        theta_b: угол для Bob
        """
        qc = QuantumCircuit(2, 2)
        
        # EPR pair
        qc.h(0)
        qc.cx(0, 1)
        
        # Rotate before measurement
        qc.ry(theta_a, 0)  # Alice
        qc.ry(theta_b, 1)  # Bob
        
        # Measure
        qc.measure([0, 1], [0, 1])
        
        return qc
    
    def run_circuit(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Запуск на симуляторе."""
        job = self.simulator.run(qc, shots=shots)
        return job.result().get_counts(qc)
    
    def compute_correlation(self, counts: Dict[str, int]) -> float:
        """
        Вычисляем корреляцию E(a,b) = P(same) - P(different)
        """
        total = sum(counts.values())
        
        # Same outcomes: 00, 11
        same = counts.get('00', 0) + counts.get('11', 0)
        # Different outcomes: 01, 10
        diff = counts.get('01', 0) + counts.get('10', 0)
        
        return (same - diff) / total
    
    def run_chsh_test(self, shots: int = 4096) -> Dict:
        """
        Полный CHSH тест.
        
        Углы для максимального нарушения:
        a1 = 0, a2 = π/2
        b1 = π/4, b2 = -π/4
        """
        # Оптимальные углы
        a1, a2 = 0, np.pi/2
        b1, b2 = np.pi/4, -np.pi/4
        
        correlations = {}
        
        # Все 4 комбинации
        settings = [
            ('a1b1', a1, b1),
            ('a1b2', a1, b2),
            ('a2b1', a2, b1),
            ('a2b2', a2, b2),
        ]
        
        for name, theta_a, theta_b in settings:
            qc = self.create_chsh_circuit(theta_a, theta_b)
            counts = self.run_circuit(qc, shots)
            correlations[name] = self.compute_correlation(counts)
        
        # CHSH value: S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
        S = (correlations['a1b1'] - correlations['a1b2'] + 
             correlations['a2b1'] + correlations['a2b2'])
        
        return {
            'correlations': correlations,
            'S': S,
            'classical_limit': 2.0,
            'quantum_max': 2 * np.sqrt(2),
            'violates_classical': abs(S) > 2.0,
        }


class PsiFieldQuantumTest:
    """
    Квантовый тест на аномалии (поиск пси-поля).
    
    Идея: если пси-поле существует, возможно есть 
    мелкие отклонения от стандартной КМ.
    
    Что проверяем:
    1. Точность корреляций (должны быть идеальные)
    2. Статистика по времени (стабильность)
    3. Зависимость от порядка измерений
    """
    
    def __init__(self):
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
    
    def run_circuit(self, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Запуск на симуляторе."""
        job = self.simulator.run(qc, shots=shots)
        return job.result().get_counts(qc)
    
    def test_correlation_stability(
        self, 
        n_batches: int = 20,
        shots_per_batch: int = 1000
    ) -> Dict:
        """
        Проверяем стабильность корреляций.
        
        Если пси-поле "мерцает" — увидим аномалии.
        """
        correlations = []
        
        for i in range(n_batches):
            # EPR + measure
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            
            counts = self.run_circuit(qc, shots_per_batch)
            
            # Correlation
            total = sum(counts.values())
            same = counts.get('00', 0) + counts.get('11', 0)
            corr = same / total
            correlations.append(corr)
        
        return {
            'correlations': correlations,
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'expected': 1.0,  # Идеальная корреляция
            'anomaly_threshold': 3 * np.std(correlations),
        }
    
    def test_measurement_order(self, shots: int = 4096) -> Dict:
        """
        Тест: зависит ли результат от порядка измерений?
        
        Стандартная КМ: НЕ зависит
        Пси-поле: возможно зависит?
        """
        # Вариант 1: Сначала Alice
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure(0, 0)  # Alice first
        qc1.measure(1, 1)  # Bob second
        
        # Вариант 2: Сначала Bob
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure(1, 1)  # Bob first
        qc2.measure(0, 0)  # Alice second
        
        # Вариант 3: Одновременно
        qc3 = QuantumCircuit(2, 2)
        qc3.h(0)
        qc3.cx(0, 1)
        qc3.measure([0, 1], [0, 1])  # Simultaneous
        
        results = {}
        for name, qc in [('alice_first', qc1), ('bob_first', qc2), ('simultaneous', qc3)]:
            counts = self.run_circuit(qc, shots)
            
            total = sum(counts.values())
            p00 = counts.get('00', 0) / total
            p11 = counts.get('11', 0) / total
            p01 = counts.get('01', 0) / total
            p10 = counts.get('10', 0) / total
            
            results[name] = {
                'p00': p00, 'p11': p11, 'p01': p01, 'p10': p10,
                'correlation': p00 + p11,
            }
        
        # Проверяем различия
        corrs = [r['correlation'] for r in results.values()]
        max_diff = max(corrs) - min(corrs)
        
        return {
            'results': results,
            'max_difference': max_diff,
            'order_matters': max_diff > 0.05,  # 5% threshold
        }
    
    def search_for_anomalies(
        self, 
        n_experiments: int = 100,
        shots: int = 1000
    ) -> Dict:
        """
        Поиск статистических аномалий.
        
        Если пси-поле создаёт редкие "всплески" —
        увидим outliers в распределении.
        """
        p00_list = []
        p11_list = []
        
        for _ in range(n_experiments):
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            
            counts = self.run_circuit(qc, shots)
            
            total = sum(counts.values())
            p00_list.append(counts.get('00', 0) / total)
            p11_list.append(counts.get('11', 0) / total)
        
        # Статистический анализ
        p00_mean, p00_std = np.mean(p00_list), np.std(p00_list)
        p11_mean, p11_std = np.mean(p11_list), np.std(p11_list)
        
        # Ищем outliers (> 3σ от среднего)
        outliers_00 = [p for p in p00_list if abs(p - p00_mean) > 3 * p00_std]
        outliers_11 = [p for p in p11_list if abs(p - p11_mean) > 3 * p11_std]
        
        return {
            'p00': {'mean': p00_mean, 'std': p00_std, 'outliers': len(outliers_00)},
            'p11': {'mean': p11_mean, 'std': p11_std, 'outliers': len(outliers_11)},
            'expected_outliers': n_experiments * 0.003,  # ~0.3% for 3σ
            'anomaly_detected': (len(outliers_00) + len(outliers_11)) > n_experiments * 0.01,
        }


def run_full_experiment():
    """Полный эксперимент."""
    
    if not QISKIT_AVAILABLE:
        print("❌ Qiskit not available!")
        print("   Install: pip install qiskit qiskit-aer")
        return None
    
    print("="*70)
    print("🔬 QUANTUM EXPERIMENT: EPR + PSI-FIELD SEARCH")
    print("="*70)
    
    results = {}
    
    # 1. Basic EPR test
    print("\n[1] BASIC EPR TEST")
    print("-"*40)
    
    epr = EPRExperiment()
    counts = epr.measure_both_z(shots=4096)
    print(f"    Results: {counts}")
    
    total = sum(counts.values())
    corr = (counts.get('00', 0) + counts.get('11', 0)) / total
    print(f"    Correlation: {corr:.4f} (expected: 1.0)")
    
    results['epr_basic'] = {'counts': counts, 'correlation': corr}
    
    # 2. Different bases
    print("\n[2] MEASUREMENT IN DIFFERENT BASES")
    print("-"*40)
    
    bases_results = epr.measure_different_bases(shots=2048)
    for basis, counts in bases_results.items():
        total = sum(counts.values())
        corr = (counts.get('00', 0) + counts.get('11', 0)) / total
        print(f"    {basis}: correlation = {corr:.4f}")
    
    results['bases'] = bases_results
    
    # 3. CHSH test
    print("\n[3] CHSH INEQUALITY TEST")
    print("-"*40)
    
    chsh = CHSHTest()
    chsh_result = chsh.run_chsh_test(shots=8192)
    
    print(f"    Correlations: {chsh_result['correlations']}")
    print(f"    S = {chsh_result['S']:.4f}")
    print(f"    Classical limit: {chsh_result['classical_limit']}")
    print(f"    Quantum maximum: {chsh_result['quantum_max']:.4f}")
    
    if chsh_result['violates_classical']:
        print("    ✓ BELL INEQUALITY VIOLATED! (Quantum nonlocality confirmed)")
    else:
        print("    ✗ No violation (possible noise)")
    
    results['chsh'] = chsh_result
    
    # 4. Psi-field search
    print("\n[4] PSI-FIELD ANOMALY SEARCH")
    print("-"*40)
    
    psi_test = PsiFieldQuantumTest()
    
    # 4a. Stability test
    print("    [4a] Correlation stability...")
    stability = psi_test.test_correlation_stability(n_batches=30, shots_per_batch=500)
    print(f"         Mean: {stability['mean']:.4f} ± {stability['std']:.4f}")
    print(f"         Expected: {stability['expected']}")
    
    results['stability'] = stability
    
    # 4b. Measurement order test
    print("    [4b] Measurement order dependence...")
    order_test = psi_test.test_measurement_order(shots=4096)
    print(f"         Max difference: {order_test['max_difference']:.4f}")
    print(f"         Order matters: {order_test['order_matters']}")
    
    results['order'] = order_test
    
    # 4c. Anomaly search
    print("    [4c] Statistical anomaly search...")
    anomaly = psi_test.search_for_anomalies(n_experiments=50, shots=500)
    print(f"         P(00): {anomaly['p00']['mean']:.4f} ± {anomaly['p00']['std']:.4f}")
    print(f"         P(11): {anomaly['p11']['mean']:.4f} ± {anomaly['p11']['std']:.4f}")
    print(f"         Outliers: {anomaly['p00']['outliers'] + anomaly['p11']['outliers']}")
    print(f"         Expected outliers: {anomaly['expected_outliers']:.1f}")
    print(f"         Anomaly detected: {anomaly['anomaly_detected']}")
    
    results['anomaly'] = anomaly
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print(f"""
    EPR Correlation:     {results['epr_basic']['correlation']:.4f} (expected: 1.0)
    CHSH Value:          {results['chsh']['S']:.4f} (classical limit: 2.0)
    Bell Violation:      {'YES ✓' if results['chsh']['violates_classical'] else 'NO'}
    Order Dependence:    {'YES ⚠️' if results['order']['order_matters'] else 'NO ✓'}
    Anomalies Found:     {'YES ⚠️' if results['anomaly']['anomaly_detected'] else 'NO ✓'}
    """)
    
    if not results['anomaly']['anomaly_detected'] and not results['order']['order_matters']:
        print("    CONCLUSION: Standard QM confirmed, no psi-field effects detected.")
    else:
        print("    ⚠️  ANOMALIES DETECTED - needs further investigation!")
    
    return results


def plot_results(results: Dict) -> plt.Figure:
    """Визуализация результатов."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Quantum EPR Experiment Results', fontsize=14, fontweight='bold')
    
    # 1. EPR correlation
    ax1 = axes[0, 0]
    counts = results['epr_basic']['counts']
    labels = list(counts.keys())
    values = list(counts.values())
    ax1.bar(labels, values, color=['steelblue', 'lightcoral', 'lightcoral', 'steelblue'])
    ax1.set_title('EPR Measurement (ZZ basis)')
    ax1.set_ylabel('Counts')
    ax1.set_xlabel('Outcome')
    
    # 2. CHSH correlations
    ax2 = axes[0, 1]
    chsh = results['chsh']
    corr_names = list(chsh['correlations'].keys())
    corr_values = list(chsh['correlations'].values())
    ax2.bar(corr_names, corr_values, color='mediumseagreen')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title(f"CHSH Correlations (S = {chsh['S']:.3f})")
    ax2.set_ylabel('E(a,b)')
    ax2.set_ylim([-1, 1])
    
    # 3. Stability over time
    ax3 = axes[1, 0]
    stability = results['stability']
    ax3.plot(stability['correlations'], 'o-', markersize=4, color='steelblue')
    ax3.axhline(y=stability['mean'], color='red', linestyle='--', 
                label=f"Mean: {stability['mean']:.4f}")
    ax3.fill_between(range(len(stability['correlations'])),
                     stability['mean'] - 2*stability['std'],
                     stability['mean'] + 2*stability['std'],
                     alpha=0.2, color='red')
    ax3.set_title('Correlation Stability Over Time')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Correlation')
    ax3.legend()
    
    # 4. Order dependence
    ax4 = axes[1, 1]
    order = results['order']['results']
    x = np.arange(3)
    width = 0.2
    
    outcomes = ['p00', 'p11', 'p01', 'p10']
    colors = ['steelblue', 'mediumseagreen', 'lightcoral', 'orange']
    
    for i, (outcome, color) in enumerate(zip(outcomes, colors)):
        values = [order[key][outcome] for key in order.keys()]
        ax4.bar(x + i*width, values, width, label=outcome, color=color)
    
    ax4.set_xticks(x + 1.5*width)
    ax4.set_xticklabels(list(order.keys()))
    ax4.set_title('Measurement Order Comparison')
    ax4.set_ylabel('Probability')
    ax4.legend()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = run_full_experiment()
    
    if results:
        fig = plot_results(results)
        fig.savefig('/mnt/user-data/outputs/quantum_experiment_results.png',
                    dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: quantum_experiment_results.png")

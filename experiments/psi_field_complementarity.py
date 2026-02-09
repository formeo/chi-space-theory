"""
Ψ-Field Final Honest Test v6.0
==============================

ЧЕСТНАЯ ПРОВЕРКА с правильными метриками.

Ключевое открытие: I_confidence ≠ which-way информация!
Детектор может быть "уверен", но если χ не коррелирует с x — информации нет.

ПРАВИЛЬНАЯ МЕТРИКА:
    I = способность предсказать сторону экрана (x > 0 или x < 0) по χ

Author: Roman Gordienko
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    L: float = 100.0
    N: int = 1024
    hbar: float = 1.0
    m: float = 1.0
    k0: float = 30.0
    sigma: float = 3.0
    x0: float = -35.0
    slit_separation: float = 10.0
    slit_width: float = 1.5
    screen_distance: float = 50.0
    
    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.k = fftfreq(self.N, self.dx) * 2 * np.pi
        self.left_center = -self.slit_separation / 2
        self.right_center = self.slit_separation / 2


class WaveFunction:
    def __init__(self, cfg):
        self.cfg = cfg
        self.psi = np.zeros(cfg.N, dtype=complex)
        self.psi_left = None
        self.psi_right = None
        self.true_path = None  # Истинный путь (для ideal detector)
        
    def initialize(self):
        x, c = self.cfg.x, self.cfg
        self.psi = np.exp(1j * c.k0 * (x - c.x0)) * np.exp(-((x - c.x0)**2) / (4 * c.sigma**2))
        self._normalize()
        
    def _normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.cfg.dx)
        if norm > 1e-10:
            self.psi /= norm
            
    def propagate(self, dt):
        c = self.cfg
        psi_k = fft(self.psi) * c.dx
        E_k = c.hbar**2 * c.k**2 / (2 * c.m)
        psi_k *= np.exp(-1j * E_k * dt / c.hbar)
        self.psi = ifft(psi_k) / c.dx
        
    def apply_slits_with_path_selection(self):
        """
        Прохождение через щели С ВЫБОРОМ ПУТИ.
        
        ФИЗИКА: Частица проходит через ОДНУ щель (сэмплируем).
        Но волновая функция остаётся когерентной суммой.
        """
        c = self.cfg
        x = c.x
        
        left_mask = (x > c.left_center - c.slit_width/2) & (x < c.left_center + c.slit_width/2)
        right_mask = (x > c.right_center - c.slit_width/2) & (x < c.right_center + c.slit_width/2)
        
        # Амплитуды в щелях
        amp_left = np.sum(np.abs(self.psi[left_mask])**2) * c.dx
        amp_right = np.sum(np.abs(self.psi[right_mask])**2) * c.dx
        total = amp_left + amp_right
        
        if total < 1e-15:
            self.true_path = np.random.choice([0, 1])
        else:
            # ИСТИННЫЙ путь — сэмплируем по амплитудам
            prob_left = amp_left / total
            self.true_path = 0 if np.random.random() < prob_left else 1
        
        # Компоненты
        self.psi_left = np.zeros_like(self.psi)
        self.psi_right = np.zeros_like(self.psi)
        self.psi_left[left_mask] = self.psi[left_mask]
        self.psi_right[right_mask] = self.psi[right_mask]
        
        # Полная ψ = сумма (интерференция!)
        self.psi = self.psi_left + self.psi_right
        self._normalize()
        
    def propagate_components(self, dt):
        c = self.cfg
        for comp in [self.psi_left, self.psi_right]:
            if comp is not None:
                psi_k = fft(comp) * c.dx
                E_k = c.hbar**2 * c.k**2 / (2 * c.m)
                psi_k *= np.exp(-1j * E_k * dt / c.hbar)
                comp[:] = ifft(psi_k) / c.dx
        self.psi = self.psi_left + self.psi_right
        
    def measure_position(self):
        prob = np.abs(self.psi)**2
        prob /= np.sum(prob)
        idx = np.random.choice(len(self.cfg.x), p=prob)
        return self.cfg.x[idx]


def run_experiment(n_particles, mode):
    """
    mode:
        'control' — без χ
        'collapse' — χ с коллапсом
        'psi_field' — χ БЕЗ коллапса (истинный путь известен!)
    """
    cfg = Config()
    
    x_data = []
    chi_data = []
    true_path_data = []
    
    v = cfg.k0 * cfg.hbar / cfg.m
    t_total = cfg.screen_distance / v
    n_steps = 400
    dt = t_total / n_steps
    
    for i in range(n_particles):
        wf = WaveFunction(cfg)
        wf.initialize()
        
        # До щелей
        for _ in range(n_steps // 3):
            wf.propagate(dt)
        
        # Щели
        wf.apply_slits_with_path_selection()
        true_path = wf.true_path
        
        if mode == 'collapse':
            # Коллапсируем на основе истинного пути
            if true_path == 0:
                wf.psi = wf.psi_left.copy()
            else:
                wf.psi = wf.psi_right.copy()
            norm = np.sqrt(np.sum(np.abs(wf.psi)**2) * cfg.dx)
            if norm > 1e-10:
                wf.psi /= norm
            # После коллапса эволюция одной компоненты
            for _ in range(2 * n_steps // 3):
                wf.propagate(dt)
        else:
            # Без коллапса — интерференция!
            for _ in range(2 * n_steps // 3):
                wf.propagate_components(dt)
        
        x_final = wf.measure_position()
        
        x_data.append(x_final)
        true_path_data.append(true_path)
        
        if mode == 'control':
            chi_data.append(-1)
        else:
            chi_data.append(true_path)  # χ = истинный путь (идеальный детектор!)
            
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_particles}")
    
    return np.array(x_data), np.array(chi_data), np.array(true_path_data)


def visibility(x_data, n_bins=80):
    hist, _ = np.histogram(x_data, bins=n_bins, range=(-25, 25))
    hist = hist[hist > 0]
    if len(hist) < 3:
        return 0.0
    return (np.max(hist) - np.min(hist)) / (np.max(hist) + np.min(hist) + 1e-10)


def which_way_info(x_data, chi_data):
    """
    ЧЕСТНАЯ which-way информация:
    Способность предсказать сторону экрана по χ.
    
    I = 2 * |P(x>0|χ=1) - 0.5|
    
    Если χ = 1 (правая щель) → частица должна быть справа (x > 0)
    Если χ = 0 (левая щель) → частица должна быть слева (x < 0)
    
    Идеальная корреляция: I = 1
    Нет корреляции: I = 0
    """
    mask = chi_data >= 0
    if np.sum(mask) < 20:
        return 0.0
    
    x = x_data[mask]
    chi = chi_data[mask]
    
    # Для χ = 0 (левая): ожидаем x < 0
    left_mask = chi == 0
    if np.sum(left_mask) > 10:
        p_correct_left = np.mean(x[left_mask] < 0)
    else:
        p_correct_left = 0.5
    
    # Для χ = 1 (правая): ожидаем x > 0
    right_mask = chi == 1
    if np.sum(right_mask) > 10:
        p_correct_right = np.mean(x[right_mask] > 0)
    else:
        p_correct_right = 0.5
    
    # Средняя точность предсказания
    accuracy = (p_correct_left + p_correct_right) / 2
    
    # Преобразуем в I ∈ [0, 1]: accuracy 0.5 → I=0, accuracy 1.0 → I=1
    I = 2 * (accuracy - 0.5)
    
    return max(0, I)  # Не может быть отрицательной


def main():
    print("=" * 80)
    print("Ψ-FIELD FINAL HONEST TEST v6.0")
    print("=" * 80)
    print("\nКлючевой вопрос: Сохраняется ли корреляция (путь ↔ позиция)")
    print("после интерференции?\n")
    
    n_particles = 2000
    
    print(f"[1/3] CONTROL: без χ-измерения")
    x_ctrl, chi_ctrl, _ = run_experiment(n_particles, 'control')
    
    print(f"\n[2/3] COLLAPSE: χ с коллапсом (стандартная КМ)")
    x_coll, chi_coll, _ = run_experiment(n_particles, 'collapse')
    
    print(f"\n[3/3] Ψ-FIELD: χ БЕЗ коллапса (идеальный детектор)")
    x_psi, chi_psi, _ = run_experiment(n_particles, 'psi_field')
    
    # Анализ
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    results = {
        'Control': (x_ctrl, chi_ctrl),
        'Collapse': (x_coll, chi_coll),
        'Ψ-Field': (x_psi, chi_psi),
    }
    
    print(f"\n{'Mode':<15} {'Visibility V':>15} {'Which-way I':>15} {'V + I':>15}")
    print("-" * 65)
    
    analysis = {}
    for name, (x, chi) in results.items():
        V = visibility(x)
        I = which_way_info(x, chi)
        analysis[name] = {'V': V, 'I': I, 'sum': V + I}
        print(f"{name:<15} {V:>15.4f} {I:>15.4f} {V + I:>15.4f}")
    
    print("-" * 65)
    print("Граница Бора: V + I ≤ 1")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    for i, (name, (x, chi)) in enumerate(results.items()):
        # Гистограмма
        ax = axes[0, i]
        ax.hist(x, bins=80, range=(-25, 25), density=True,
               alpha=0.75, color=colors[i], edgecolor='black', linewidth=0.3)
        V = analysis[name]['V']
        ax.set_title(f"{name}\nV = {V:.4f}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Position x")
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.3)
        
        # Scatter x vs chi
        ax = axes[1, i]
        if name == 'Control':
            ax.text(0.5, 0.5, "No χ measurement", 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
        else:
            mask = chi >= 0
            jitter = np.random.uniform(-0.1, 0.1, np.sum(mask))
            ax.scatter(x[mask], chi[mask] + jitter, alpha=0.3, s=8, color=colors[i])
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            
            I = analysis[name]['I']
            ax.set_title(f"Which-way I = {I:.4f}", fontsize=11)
        
        ax.set_xlabel("Position x")
        ax.set_ylabel("χ (0=left, 1=right)")
        ax.set_ylim([-0.3, 1.3])
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Добавляем summary bar chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    modes = list(analysis.keys())
    x_pos = np.arange(len(modes))
    V_vals = [analysis[m]['V'] for m in modes]
    I_vals = [analysis[m]['I'] for m in modes]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, V_vals, width, label='Visibility V', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, I_vals, width, label='Which-way I', color='coral', alpha=0.8)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Bohr limit')
    
    for i, m in enumerate(modes):
        total = V_vals[i] + I_vals[i]
        color = 'green' if total > 1.0 else 'black'
        ax.text(i, max(V_vals[i], I_vals[i]) + 0.05, 
               f"Σ = {total:.3f}", ha='center', fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim([0, 1.2])
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title("COMPLEMENTARITY TEST: V + I ≤ 1 ?", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Вердикт
    print("\n" + "=" * 80)
    print("ВЕРДИКТ")
    print("=" * 80)
    
    psi = analysis['Ψ-Field']
    coll = analysis['Collapse']
    
    print(f"\nСтандартная КМ (collapse):")
    print(f"  V = {coll['V']:.4f} (интерференция разрушена)")
    print(f"  I = {coll['I']:.4f} (путь известен)")
    print(f"  V + I = {coll['sum']:.4f}")
    
    print(f"\nΨ-Field (без коллапса):")
    print(f"  V = {psi['V']:.4f} (интерференция сохранена)")
    print(f"  I = {psi['I']:.4f} (корреляция путь↔позиция)")
    print(f"  V + I = {psi['sum']:.4f}")
    
    if psi['sum'] > 1.02:
        print("\n🔥🔥🔥 НАРУШЕНИЕ КОМПЛЕМЕНТАРНОСТИ! 🔥🔥🔥")
        print("Ψ-поле ВОЗМОЖНО существует!")
        verdict = "VIOLATION"
    elif psi['sum'] < 0.98:
        print("\n✗ КОМПЛЕМЕНТАРНОСТЬ СОБЛЮДАЕТСЯ")
        print("\nИнтерференция УНИЧТОЖАЕТ корреляцию путь↔позиция!")
        print("Даже зная истинный путь, мы НЕ МОЖЕМ предсказать")
        print("на какую сторону экрана попадёт частица.")
        print("\nΨ-поле в данной формулировке НЕВОЗМОЖНО.")
        verdict = "NO_VIOLATION"
    else:
        print("\n⚠️ ГРАНИЧНЫЙ РЕЗУЛЬТАТ (V + I ≈ 1)")
        verdict = "MARGINAL"
    
    # Сохранение
    fig.savefig('/mnt/user-data/outputs/psi_field_final_distributions.png', dpi=300, bbox_inches='tight')
    fig2.savefig('/mnt/user-data/outputs/psi_field_final_summary.png', dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Сохранено: psi_field_final_distributions.png")
    print(f"✓ Сохранено: psi_field_final_summary.png")
    
    return analysis, verdict


if __name__ == "__main__":
    analysis, verdict = main()

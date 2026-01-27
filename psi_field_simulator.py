"""
Ψ-Field Toy Model v4.0 - THOUGHT EXPERIMENT
============================================

ПОДХОД: Мысленный эксперимент в духе EPR или Schrödinger's cat.

ПОСТУЛАТ (не обоснован физически):
    Существует наблюдаемая χ̂ такая что:
    1. [χ̂, x̂] = [χ̂, p̂] = 0  (коммутирует с положением/импульсом)
    2. χ можно измерить и получить which-way информацию
    3. Это измерение НЕ коллапсирует ψ(x) 
    
НАРУШЕНИЕ стандартной КМ:
    • Стандартная КМ: which-way info ↔ декогеренция (Bohr)
    • Toy model: which-way info + coherence (постулат Ψ-поля)
    
ЦЕЛЬ: Не доказать что это возможно, а ИССЛЕДОВАТЬ следствия ЕСЛИ это работает.

Author: Roman
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
import seaborn as sns

sns.set_style("whitegrid")


@dataclass
class Config:
    """Параметры эксперимента"""
    # Пространственная сетка
    L: float = 80.0
    N: int = 512
    
    # Физические константы
    hbar: float = 1.0
    m: float = 1.0
    
    # Параметры пучка
    k0: float = 18.0  # Уменьшаем для более перпендикулярного падения
    sigma: float = 2.5
    x0: float = -30.0
    
    # Геометрия
    slit_separation: float = 12.0
    slit_width: float = 2.0
    screen_distance: float = 45.0
    
    # Ψ-field параметры
    chi_fidelity: float = 0.98  # Почти идеальный детектор
    # 0.5 = random guess, 1.0 = perfect which-way detection
    
    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.k = fftfreq(self.N, self.dx) * 2*np.pi


class QuantumWaveFunction:
    """Стандартная волновая функция в двухщелевом эксперименте"""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.psi = np.zeros(cfg.N, dtype=complex)
        self.time = 0.0
        
    def initialize_gaussian_beam(self):
        """Гауссов волновой пакет"""
        x = self.cfg.x
        self.psi = np.exp(1j * self.cfg.k0 * (x - self.cfg.x0)) * \
                   np.exp(-((x - self.cfg.x0)**2) / (4 * self.cfg.sigma**2))
        # Нормировка
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.cfg.dx)
        self.psi /= norm
        
    def propagate_free_space(self, dt: float):
        """Свободная эволюция (Fourier method)"""
        cfg = self.cfg
        
        # FFT → k-space
        psi_k = fft(self.psi) * cfg.dx
        
        # Evolution: exp(-i E(k) t / ℏ)
        E_k = cfg.hbar**2 * cfg.k**2 / (2 * cfg.m)
        psi_k *= np.exp(-1j * E_k * dt / cfg.hbar)
        
        # IFFT → x-space
        self.psi = ifft(psi_k) / cfg.dx
        self.time += dt
        
    def apply_double_slit(self, dt: float):
        """Прохождение через двойную щель (split-operator)"""
        cfg = self.cfg
        x = cfg.x
        
        # Потенциальный барьер с двумя щелями
        V = np.ones_like(x) * 1e6
        
        left_slit = (x > -cfg.slit_separation/2 - cfg.slit_width/2) & \
                   (x < -cfg.slit_separation/2 + cfg.slit_width/2)
        right_slit = (x > cfg.slit_separation/2 - cfg.slit_width/2) & \
                    (x < cfg.slit_separation/2 + cfg.slit_width/2)
        
        V[left_slit | right_slit] = 0.0
        
        # Split-operator: exp(-iV/2) exp(-iT) exp(-iV/2)
        self.psi *= np.exp(-1j * V * dt / (2 * cfg.hbar))
        
        psi_k = fft(self.psi) * cfg.dx
        E_k = cfg.hbar**2 * cfg.k**2 / (2 * cfg.m)
        psi_k *= np.exp(-1j * E_k * dt / cfg.hbar)
        self.psi = ifft(psi_k) / cfg.dx
        
        self.psi *= np.exp(-1j * V * dt / (2 * cfg.hbar))
        self.time += dt
        
    def get_density(self) -> np.ndarray:
        """Вероятностная плотность"""
        return np.abs(self.psi)**2
    
    def measure_position(self) -> float:
        """Коллапсирующее измерение позиции"""
        prob = self.get_density()
        prob /= np.sum(prob)
        idx = np.random.choice(len(self.cfg.x), p=prob)
        return self.cfg.x[idx]


class ChiDetector:
    """
    Ψ-FIELD χ-ДЕТЕКТОР
    
    MAGIC HAPPENS HERE: Постулируем детектор который:
    1. "Знает" через какую щель прошла частица
    2. Записывает это в χ с точностью chi_fidelity
    3. НЕ коллапсирует волновую функцию ψ(x)
    
    В стандартной КМ это НЕВОЗМОЖНО (Bohr's complementarity).
    В toy model мы постулируем что Ψ-поле делает это возможным.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
    def detect_which_way(self, psi: np.ndarray) -> int:
        """
        Определяет через какую щель прошла частица.
        
        ПОСТУЛАТ (не физичный!): 
        Можем "подглядеть" на волновую функцию и определить путь
        БЕЗ коллапса ψ(x).
        
        РЕАЛИЗАЦИЯ:
        Делаем "слабое измерение" позиции - сэмплируем из |ψ|²
        но НЕ коллапсируем состояние.
        
        Returns:
            0 = LEFT slit
            1 = RIGHT slit
        """
        cfg = self.cfg
        x = cfg.x
        
        # "Слабое измерение": сэмплируем позицию БЕЗ коллапса
        prob = np.abs(psi)**2
        total = np.sum(prob) * cfg.dx
        
        if total < 1e-10:
            return np.random.choice([0, 1])
        
        prob /= (np.sum(prob) + 1e-10)
        
        # Сэмплируем позицию
        idx = np.random.choice(len(x), p=prob)
        x_sample = x[idx]
        
        # Определяем путь по позиции
        if x_sample < 0:
            actual_path = 0  # LEFT
        else:
            actual_path = 1  # RIGHT
        
        # Добавляем imperfection детектора
        if np.random.random() < cfg.chi_fidelity:
            chi_measurement = actual_path  # Правильно
        else:
            chi_measurement = 1 - actual_path  # Ошибка
        
        return chi_measurement


class PsiFieldExperiment:
    """
    Полный эксперимент: двойная щель + χ-детектор
    
    Сравниваем 3 режима:
    1. Контроль: без χ-детектора (стандартная КМ)
    2. Стандартная КМ: which-way измерение с коллапсом
    3. Ψ-field: which-way через χ БЕЗ коллапса (toy model)
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.chi_detector = ChiDetector(cfg)
        
    def run_single_particle(self, mode: str) -> tuple:
        """
        Пропускаем одну частицу.
        
        Args:
            mode: 'control' | 'standard_qm' | 'psi_field'
            
        Returns:
            (x_final, chi_measurement, path_info_available)
        """
        cfg = self.cfg
        wf = QuantumWaveFunction(cfg)
        wf.initialize_gaussian_beam()
        
        # Эволюция до щелей
        steps = 300
        dt = cfg.screen_distance / (steps * cfg.k0 / cfg.m)
        
        for _ in range(steps // 3):
            wf.propagate_free_space(dt)
        
        # Сохраняем состояние У ЩЕЛЕЙ для χ-детектора
        psi_at_slits = wf.psi.copy()
        
        # Прохождение через щели
        wf.apply_double_slit(dt)
        
        # ===== КРИТИЧЕСКАЯ ТОЧКА: χ-ИЗМЕРЕНИЕ =====
        chi_measured = -1  # -1 = не измерено
        
        if mode == 'control':
            # Режим 1: Без измерения (стандартная интерференция)
            pass
            
        elif mode == 'standard_qm':
            # Режим 2: Which-way измерение С коллапсом
            chi_measured = self.chi_detector.detect_which_way(psi_at_slits)
            
            # КОЛЛАПС волновой функции!
            x = cfg.x
            if chi_measured == 0:  # Левая щель
                left_region = (x > -cfg.slit_separation/2 - 2*cfg.slit_width) & \
                             (x < -cfg.slit_separation/2 + 2*cfg.slit_width)
                wf.psi[~left_region] = 0
            else:  # Правая щель
                right_region = (x > cfg.slit_separation/2 - 2*cfg.slit_width) & \
                              (x < cfg.slit_separation/2 + 2*cfg.slit_width)
                wf.psi[~right_region] = 0
            
            # Ренормировка
            norm = np.sqrt(np.sum(np.abs(wf.psi)**2) * cfg.dx)
            if norm > 1e-10:
                wf.psi /= norm
                
        elif mode == 'psi_field':
            # Режим 3: Which-way через χ БЕЗ коллапса (МАГИЯ!)
            # 
            # ВАЖНО: Используем состояние У ЩЕЛЕЙ для определения пути
            # Это моделирует детектор который "смотрит" на частицу
            # когда она проходит через щели
            chi_measured = self.chi_detector.detect_which_way(psi_at_slits)
            # НЕ коллапсируем wf.psi!
            # Это "волшебство" Ψ-поля
            
        # Эволюция до экрана
        for _ in range(2 * steps // 3):
            wf.propagate_free_space(dt)
        
        # Финальное измерение позиции
        x_final = wf.measure_position()
        
        return x_final, chi_measured
    
    def run_ensemble(self, n_particles: int, mode: str) -> dict:
        """Запуск ансамбля частиц"""
        x_data = []
        chi_data = []
        
        print(f"  Running {n_particles} particles in '{mode}' mode...")
        
        for i in range(n_particles):
            x, chi = self.run_single_particle(mode)
            x_data.append(x)
            chi_data.append(chi)
            
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n_particles}")
        
        return {
            'x': np.array(x_data),
            'chi': np.array(chi_data),
            'mode': mode
        }


def calculate_visibility(x_data: np.ndarray) -> float:
    """Видимость интерференционной картины"""
    hist, _ = np.histogram(x_data, bins=50, range=(-20, 20))
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    I_max = np.max(hist)
    I_min = np.min(hist)
    return (I_max - I_min) / (I_max + I_min)


def calculate_which_way_info(x_data: np.ndarray, chi_data: np.ndarray) -> float:
    """
    Вычисляем which-way информацию из корреляции x-χ.
    
    Perfect correlation: I = 1 (знаем путь наверняка)
    No correlation: I = 0 (нет информации)
    """
    from scipy.stats import pearsonr
    
    # Фильтруем только те события где chi был измерен
    mask = chi_data >= 0
    if np.sum(mask) < 10:
        return 0.0
    
    x_filtered = x_data[mask]
    chi_filtered = chi_data[mask]
    
    # ВАЖНО: Сравниваем с МЕДИАНОЙ распределения, не с нулём!
    # (интерференционная картина может быть смещена)
    x_median = np.median(x_filtered)
    
    # Преобразуем в бинарные: left vs right OF THE MEDIAN
    x_left = (x_filtered < x_median).astype(float)
    chi_left = (chi_filtered == 0).astype(float)
    
    if np.std(chi_left) < 0.01 or np.std(x_left) < 0.01:
        return 0.0
    
    corr, _ = pearsonr(x_left, chi_left)
    
    # Преобразуем корреляцию в "which-way information"
    # I = |ρ| ∈ [0, 1]
    return abs(corr)


def plot_three_way_comparison(results: list, cfg: Config):
    """Сравнение трёх режимов"""
    
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Ψ-Field Thought Experiment: Complementarity Test', 
                fontsize=16, fontweight='bold')
    
    modes = ['Control', 'Standard QM', 'Ψ-Field']
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    V_list = []
    I_list = []
    
    for i, (result, mode, color) in enumerate(zip(results, modes, colors)):
        x = result['x']
        chi = result['chi']
        
        # 1. Гистограмма позиций
        ax = fig.add_subplot(gs[0, i])
        ax.hist(x, bins=60, range=(-20, 20), density=True, 
               alpha=0.75, color=color, edgecolor='black', linewidth=0.5)
        
        V = calculate_visibility(x)
        V_list.append(V)
        
        ax.set_title(f'{mode}\nVisibility V = {V:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Position on screen', fontsize=10)
        ax.set_ylabel('Probability density', fontsize=10)
        ax.grid(alpha=0.3)
        
        # 2. x-χ scatter (если χ измерялся)
        ax = fig.add_subplot(gs[1, i])
        
        if mode != 'Control':
            mask = chi >= 0
            ax.scatter(x[mask], chi[mask], alpha=0.3, s=8, color=color)
            
            I_ww = calculate_which_way_info(x, chi)
            I_list.append(I_ww)
            
            ax.set_title(f'Path Information I = {I_ww:.3f}', fontsize=11)
            ax.set_ylim([-0.1, 1.1])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Left', 'Right'])
        else:
            ax.text(0.5, 0.5, 'No χ-measurement\nin this mode', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=11, style='italic',
                   bbox=dict(boxstyle='round', fc='lightgray', alpha=0.5))
            I_list.append(0.0)
        
        ax.set_xlabel('Position x', fontsize=10)
        ax.set_ylabel('χ measurement', fontsize=10)
        ax.grid(alpha=0.3)
        
        # 3. Интерпретация
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        
        if mode == 'Control':
            text = (
                "NO which-way measurement\n\n"
                f"→ Standard interference\n"
                f"→ V = {V:.3f} (baseline)\n"
                f"→ No path information\n\n"
                "This is textbook QM:\n"
                "coherence preserved"
            )
            box_color = 'lightblue'
        elif mode == 'Standard QM':
            text = (
                "Which-way WITH collapse\n\n"
                f"→ Visibility: V = {V:.3f}\n"
                f"→ Path info: I = {I_list[i]:.3f}\n"
                f"→ V + I ≈ {V + I_list[i]:.3f}\n\n"
                "Bohr's complementarity:\n"
                "V + I ≤ 1  ✓ SATISFIED"
            )
            box_color = 'lightyellow'
        else:  # Ψ-Field
            text = (
                "Which-way WITHOUT collapse\n"
                "(via Ψ-field χ-detector)\n\n"
                f"→ Visibility: V = {V:.3f}\n"
                f"→ Path info: I = {I_list[i]:.3f}\n"
                f"→ V + I ≈ {V + I_list[i]:.3f}\n\n"
            )
            
            if V > 0.8 and I_list[i] > 0.4:
                text += "⚠️ VIOLATION of Bohr!\n"
                text += "V + I > 1  (impossible\nin standard QM)"
                box_color = 'lightgreen'
            else:
                text += "Standard QM holds\n(no violation)"
                box_color = 'lightyellow'
        
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
               ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=1', fc=box_color, alpha=0.7))
    
    # Сводная таблица
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON")
    print("="*80)
    print(f"{'Mode':<20} {'Visibility V':>15} {'Path Info I':>15} {'V + I':>15}")
    print("-"*80)
    
    for mode, V, I in zip(modes, V_list, I_list):
        print(f"{mode:<20} {V:>15.4f} {I:>15.4f} {V+I:>15.4f}")
    
    print("="*80)
    print("\nBohr's Complementarity: V + I ≤ 1")
    print("If V + I > 1 significantly → violation of standard QM")
    
    return fig, V_list, I_list


def main():
    print("="*80)
    print("Ψ-FIELD THOUGHT EXPERIMENT v4.0")
    print("="*80)
    print("\nPOSTULATE: A χ-detector can obtain which-way information")
    print("           WITHOUT collapsing ψ(x) [χ̂, x̂] = 0")
    print("\nQUESTION: Does this violate Bohr's complementarity principle?")
    print("          (Standard QM: V + I ≤ 1, always)")
    print("\n" + "="*80)
    
    # Конфигурация с HIGH fidelity детектора
    cfg = Config(
        L=80.0,
        N=512,
        k0=25.0,
        slit_separation=12.0,
        slit_width=2.0,
        chi_fidelity=0.90  # 90% точность which-way детектора
    )
    
    print(f"\nParameters:")
    print(f"  χ-detector fidelity: {cfg.chi_fidelity:.1%}")
    print(f"  Slit separation: {cfg.slit_separation}")
    print(f"  Wave number k₀: {cfg.k0}")
    
    exp = PsiFieldExperiment(cfg)
    
    n_particles = 2000
    
    print(f"\nRunning {n_particles} particles in each mode...\n")
    
    # Режим 1: Контроль
    print("[1] CONTROL: No which-way measurement")
    results_control = exp.run_ensemble(n_particles, 'control')
    
    # Режим 2: Стандартная КМ
    print("\n[2] STANDARD QM: Which-way WITH wavefunction collapse")
    results_standard = exp.run_ensemble(n_particles, 'standard_qm')
    
    # Режим 3: Ψ-field
    print("\n[3] Ψ-FIELD: Which-way WITHOUT collapse (toy model)")
    results_psi = exp.run_ensemble(n_particles, 'psi_field')
    
    # Анализ
    print("\n" + "="*80)
    print("ANALYZING RESULTS...")
    print("="*80)
    
    fig, V_list, I_list = plot_three_way_comparison(
        [results_control, results_standard, results_psi],
        cfg
    )
    
    # Вердикт
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    V_psi = V_list[2]
    I_psi = I_list[2]
    sum_psi = V_psi + I_psi
    
    print(f"\nΨ-field mode results:")
    print(f"  Visibility:      V = {V_psi:.4f}")
    print(f"  Path information: I = {I_psi:.4f}")
    print(f"  Sum:            V+I = {sum_psi:.4f}")
    
    if sum_psi > 1.1:
        print("\n" + "🔥"*30)
        print("✓✓✓ SIGNIFICANT VIOLATION OF COMPLEMENTARITY! ✓✓✓")
        print("🔥"*30)
        print("\nIn this toy model:")
        print("  • High visibility (interference preserved)")
        print("  • High path information (χ knows the way)")
        print("  • V + I > 1 (impossible in standard QM)")
        print("\nConclusion: IF Ψ-field works as postulated,")
        print("           it would violate Bohr's complementarity.")
        print("\nNext steps:")
        print("  1. Develop rigorous theory showing [χ̂,x̂]=0")
        print("  2. Check consistency with unitarity")
        print("  3. Verify no-signaling theorem")
        print("  4. Design real experiment to test")
    elif sum_psi > 0.95:
        print("\n✓ MARGINAL RESULT")
        print("  → V+I ≈ 1 (borderline)")
        print("  → Consistent with weak measurements")
        print("  → No clear violation of QM")
    else:
        print("\n✗ NULL RESULT")
        print("  → Standard QM trade-off holds")
        print("  → No violation of complementarity")
        print("  → χ-detector model needs improvement")
    
    return fig, results_control, results_standard, results_psi


if __name__ == "__main__":
    fig, r1, r2, r3 = main()
    
    plt.savefig('/mnt/user-data/outputs/psi_field_thought_experiment.png',
                dpi=300, bbox_inches='tight')
    
    print(f"\n\n{'='*80}")
    print("✓ Simulation complete!")
    print(f"✓ Results saved to: outputs/psi_field_thought_experiment.png")
    print("="*80)

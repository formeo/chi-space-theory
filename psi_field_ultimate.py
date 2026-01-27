"""
Ψ-Field Ultimate Test v5.0
==========================

ЦЕЛЬ: Максимально честно проверить — возможно ли V + I > 1?

Добавляем ВСЕ возможные допущения в пользу Ψ-поля:
1. Идеальный χ-детектор (fidelity = 100%)
2. Несколько методов измерения which-way информации
3. Несколько метрик для I (Pearson, Mutual Info, Weak Value)
4. Разные моменты χ-измерения (у щелей, в полёте, на экране)

Если даже так V + I ≤ 1 — комплементарность фундаментальна.
Если V + I > 1 — нашли лазейку!

Author: Roman Gordienko
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr, entropy
from dataclasses import dataclass, field
from typing import Callable
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    """Параметры эксперимента."""
    # Сетка
    L: float = 100.0
    N: int = 1024  # Увеличиваем разрешение
    
    # Физика
    hbar: float = 1.0
    m: float = 1.0
    
    # Волновой пакет
    k0: float = 30.0  # Высокий импульс для чёткой интерференции
    sigma: float = 3.0
    x0: float = -35.0
    
    # Геометрия щелей
    slit_separation: float = 10.0
    slit_width: float = 1.5
    screen_distance: float = 50.0
    
    # χ-детектор
    chi_fidelity: float = 1.0  # ИДЕАЛЬНЫЙ детектор
    
    def __post_init__(self):
        self.dx = self.L / self.N
        self.x = np.linspace(-self.L/2, self.L/2, self.N)
        self.k = fftfreq(self.N, self.dx) * 2 * np.pi
        
        # Позиции щелей
        self.left_slit_center = -self.slit_separation / 2
        self.right_slit_center = self.slit_separation / 2


class WaveFunction:
    """Волновая функция с расширенной диагностикой."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.psi = np.zeros(cfg.N, dtype=complex)
        self.psi_left = None   # Компонента через левую щель
        self.psi_right = None  # Компонента через правую щель
        
    def initialize(self):
        """Гауссов волновой пакет."""
        x = self.cfg.x
        c = self.cfg
        self.psi = np.exp(1j * c.k0 * (x - c.x0)) * \
                   np.exp(-((x - c.x0)**2) / (4 * c.sigma**2))
        self._normalize()
        
    def _normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.cfg.dx)
        if norm > 1e-10:
            self.psi /= norm
            
    def propagate(self, dt: float):
        """Свободная эволюция."""
        c = self.cfg
        psi_k = fft(self.psi) * c.dx
        E_k = c.hbar**2 * c.k**2 / (2 * c.m)
        psi_k *= np.exp(-1j * E_k * dt / c.hbar)
        self.psi = ifft(psi_k) / c.dx
        
    def apply_slits(self):
        """
        Прохождение через щели С РАЗДЕЛЕНИЕМ на компоненты.
        
        КЛЮЧЕВОЙ МОМЕНТ: сохраняем psi_left и psi_right отдельно!
        Это позволяет отслеживать "путь" без коллапса.
        """
        c = self.cfg
        x = c.x
        
        # Маски щелей
        left_mask = (x > c.left_slit_center - c.slit_width/2) & \
                   (x < c.left_slit_center + c.slit_width/2)
        right_mask = (x > c.right_slit_center - c.slit_width/2) & \
                    (x < c.right_slit_center + c.slit_width/2)
        
        # Разделяем на компоненты
        self.psi_left = np.zeros_like(self.psi)
        self.psi_right = np.zeros_like(self.psi)
        
        self.psi_left[left_mask] = self.psi[left_mask]
        self.psi_right[right_mask] = self.psi[right_mask]
        
        # Полная волновая функция = сумма (для интерференции)
        self.psi = self.psi_left + self.psi_right
        self._normalize()
        
        # Нормируем компоненты
        norm_l = np.sum(np.abs(self.psi_left)**2) * c.dx
        norm_r = np.sum(np.abs(self.psi_right)**2) * c.dx
        total = norm_l + norm_r
        
        if total > 1e-10:
            self.psi_left /= np.sqrt(total)
            self.psi_right /= np.sqrt(total)
            
    def propagate_components(self, dt: float):
        """Эволюция компонент отдельно (для отслеживания пути)."""
        c = self.cfg
        
        for comp in [self.psi_left, self.psi_right]:
            if comp is not None:
                psi_k = fft(comp) * c.dx
                E_k = c.hbar**2 * c.k**2 / (2 * c.m)
                psi_k *= np.exp(-1j * E_k * dt / c.hbar)
                comp[:] = ifft(psi_k) / c.dx
                
        # Обновляем полную ψ
        self.psi = self.psi_left + self.psi_right
        
    def get_path_amplitudes(self, x_pos: float) -> tuple[float, float]:
        """
        Амплитуды компонент в точке x.
        
        Возвращает: (|ψ_left(x)|², |ψ_right(x)|²)
        """
        idx = np.argmin(np.abs(self.cfg.x - x_pos))
        
        amp_left = np.abs(self.psi_left[idx])**2 if self.psi_left is not None else 0
        amp_right = np.abs(self.psi_right[idx])**2 if self.psi_right is not None else 0
        
        return amp_left, amp_right
    
    def measure_position(self) -> float:
        """Измерение позиции (сэмплинг из |ψ|²)."""
        prob = np.abs(self.psi)**2
        prob /= np.sum(prob)
        idx = np.random.choice(len(self.cfg.x), p=prob)
        return self.cfg.x[idx]


class PsiFieldDetector:
    """
    Ψ-Field χ-детектор с разными стратегиями.
    
    Постулат: [χ̂, x̂] = 0, но χ несёт информацию о пути.
    
    Стратегии:
    1. amplitude — по амплитудам компонент ψ_left, ψ_right
    2. weak — слабое измерение (сэмпл без коллапса)
    3. bayesian — байесовский вывод из позиции
    4. ideal — идеальное знание пути (верхняя граница)
    """
    
    def __init__(self, cfg: Config, strategy: str = "amplitude"):
        self.cfg = cfg
        self.strategy = strategy
        
    def measure_which_way(
        self, 
        wf: WaveFunction, 
        x_measured: float | None = None
    ) -> tuple[int, float]:
        """
        Измеряет через какую щель прошла частица.
        
        Returns:
            (chi, confidence): 
                chi = 0 (left) или 1 (right)
                confidence ∈ [0.5, 1.0]
        """
        if self.strategy == "amplitude":
            return self._amplitude_method(wf, x_measured)
        elif self.strategy == "weak":
            return self._weak_method(wf)
        elif self.strategy == "bayesian":
            return self._bayesian_method(wf, x_measured)
        elif self.strategy == "ideal":
            return self._ideal_method(wf, x_measured)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
    def _amplitude_method(self, wf: WaveFunction, x_pos: float) -> tuple[int, float]:
        """
        Определяем путь по амплитудам ψ_left и ψ_right в точке измерения.
        
        ДОПУЩЕНИЕ: Ψ-поле "видит" раздельные амплитуды компонент.
        """
        if wf.psi_left is None:
            return np.random.choice([0, 1]), 0.5
            
        amp_l, amp_r = wf.get_path_amplitudes(x_pos)
        total = amp_l + amp_r
        
        if total < 1e-15:
            return np.random.choice([0, 1]), 0.5
            
        prob_left = amp_l / total
        
        # χ = более вероятный путь
        chi = 0 if prob_left > 0.5 else 1
        confidence = max(prob_left, 1 - prob_left)
        
        # Добавляем fidelity детектора
        if np.random.random() > self.cfg.chi_fidelity:
            chi = 1 - chi
            
        return chi, confidence
    
    def _weak_method(self, wf: WaveFunction) -> tuple[int, float]:
        """
        Слабое измерение: сэмплируем из |ψ|² у щелей, но не коллапсируем.
        """
        c = self.cfg
        
        # Сэмплируем позицию в области щелей
        slit_region = (c.x > c.left_slit_center - 2*c.slit_width) & \
                     (c.x < c.right_slit_center + 2*c.slit_width)
        
        prob = np.abs(wf.psi)**2
        prob[~slit_region] = 0
        
        if np.sum(prob) < 1e-15:
            return np.random.choice([0, 1]), 0.5
            
        prob /= np.sum(prob)
        idx = np.random.choice(len(c.x), p=prob)
        x_sample = c.x[idx]
        
        # Определяем путь по позиции сэмпла
        chi = 0 if x_sample < 0 else 1
        
        # Confidence пропорционален удалению от центра
        confidence = min(1.0, 0.5 + abs(x_sample) / c.slit_separation)
        
        return chi, confidence
    
    def _bayesian_method(self, wf: WaveFunction, x_measured: float) -> tuple[int, float]:
        """
        Байесовский вывод: P(left|x) по теореме Байеса.
        
        P(left|x) = P(x|left) * P(left) / P(x)
        """
        if wf.psi_left is None or x_measured is None:
            return np.random.choice([0, 1]), 0.5
            
        amp_l, amp_r = wf.get_path_amplitudes(x_measured)
        total = amp_l + amp_r
        
        if total < 1e-15:
            return np.random.choice([0, 1]), 0.5
            
        # P(left|x) = |ψ_left(x)|² / (|ψ_left(x)|² + |ψ_right(x)|²)
        prob_left = amp_l / total
        
        chi = 0 if prob_left > 0.5 else 1
        confidence = max(prob_left, 1 - prob_left)
        
        return chi, confidence
    
    def _ideal_method(self, wf: WaveFunction, x_measured: float) -> tuple[int, float]:
        """
        ИДЕАЛЬНЫЙ детектор: сэмплируем путь с точной вероятностью.
        
        МАКСИМАЛЬНОЕ ДОПУЩЕНИЕ в пользу Ψ-поля!
        """
        if wf.psi_left is None:
            return np.random.choice([0, 1]), 0.5
            
        # Интегральные амплитуды (не в точке, а по всему пространству)
        amp_l = np.sum(np.abs(wf.psi_left)**2) * self.cfg.dx
        amp_r = np.sum(np.abs(wf.psi_right)**2) * self.cfg.dx
        total = amp_l + amp_r
        
        if total < 1e-15:
            return np.random.choice([0, 1]), 0.5
            
        prob_left = amp_l / total
        
        # Сэмплируем путь из распределения
        chi = 0 if np.random.random() < prob_left else 1
        confidence = max(prob_left, 1 - prob_left)
        
        return chi, confidence


class ComplementarityExperiment:
    """
    Полный эксперимент для проверки V + I ≤ 1.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
    def run_particle(
        self, 
        mode: str,
        chi_strategy: str = "amplitude"
    ) -> dict:
        """
        Один прогон частицы.
        
        Modes:
        - control: без χ-измерения
        - collapse: χ-измерение с коллапсом (стандартная КМ)
        - psi_field: χ-измерение БЕЗ коллапса
        """
        c = self.cfg
        wf = WaveFunction(c)
        wf.initialize()
        
        # Параметры эволюции
        v = c.k0 * c.hbar / c.m
        t_total = c.screen_distance / v
        n_steps = 400
        dt = t_total / n_steps
        
        # 1. Долетаем до щелей
        for _ in range(n_steps // 3):
            wf.propagate(dt)
            
        # 2. Проходим через щели (с разделением на компоненты)
        wf.apply_slits()
        
        # 3. Эволюция до экрана
        for _ in range(2 * n_steps // 3):
            wf.propagate_components(dt)
            
        # 4. Измеряем позицию на экране
        x_final = wf.measure_position()
        
        # 5. χ-измерение
        chi = -1
        chi_confidence = 0.0
        amp_left, amp_right = 0.0, 0.0
        
        if mode == "control":
            pass
            
        elif mode == "collapse":
            # Стандартная КМ: измерение → коллапс
            detector = PsiFieldDetector(c, chi_strategy)
            chi, chi_confidence = detector.measure_which_way(wf, x_final)
            
            # КОЛЛАПС: оставляем только одну компоненту
            # (но это происходит ПОСЛЕ финального измерения, 
            # так что на x_final не влияет)
            
        elif mode == "psi_field":
            # Ψ-поле: измерение БЕЗ коллапса
            detector = PsiFieldDetector(c, chi_strategy)
            chi, chi_confidence = detector.measure_which_way(wf, x_final)
            amp_left, amp_right = wf.get_path_amplitudes(x_final)
            
        return {
            "x": x_final,
            "chi": chi,
            "chi_confidence": chi_confidence,
            "amp_left": amp_left,
            "amp_right": amp_right,
        }
    
    def run_ensemble(
        self, 
        n_particles: int, 
        mode: str,
        chi_strategy: str = "amplitude"
    ) -> dict:
        """Запуск ансамбля."""
        results = {
            "x": [],
            "chi": [],
            "chi_confidence": [],
            "amp_left": [],
            "amp_right": [],
        }
        
        for i in range(n_particles):
            r = self.run_particle(mode, chi_strategy)
            for key in results:
                results[key].append(r[key])
                
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n_particles}")
                
        for key in results:
            results[key] = np.array(results[key])
            
        return results


# ==================== МЕТРИКИ ====================

def visibility(x_data: np.ndarray, n_bins: int = 80) -> float:
    """
    Видимость интерференционной картины.
    
    V = (I_max - I_min) / (I_max + I_min)
    """
    hist, _ = np.histogram(x_data, bins=n_bins, range=(-25, 25))
    
    # Убираем нули
    hist = hist[hist > 0]
    if len(hist) < 3:
        return 0.0
        
    # Находим экстремумы (для интерференционной картины)
    I_max = np.max(hist)
    I_min = np.min(hist)
    
    return (I_max - I_min) / (I_max + I_min + 1e-10)


def which_way_pearson(x_data: np.ndarray, chi_data: np.ndarray) -> float:
    """
    Which-way информация через корреляцию Пирсона.
    
    I = |corr(x > median, chi)|
    """
    mask = chi_data >= 0
    if np.sum(mask) < 20:
        return 0.0
        
    x = x_data[mask]
    chi = chi_data[mask]
    
    # Бинаризуем x относительно медианы
    x_binary = (x > np.median(x)).astype(float)
    chi_binary = (chi == 1).astype(float)
    
    if np.std(x_binary) < 0.01 or np.std(chi_binary) < 0.01:
        return 0.0
        
    corr, _ = pearsonr(x_binary, chi_binary)
    return abs(corr)


def which_way_mutual_info(x_data: np.ndarray, chi_data: np.ndarray) -> float:
    """
    Which-way информация через Mutual Information.
    
    I(X; χ) = H(X) + H(χ) - H(X, χ)
    
    Нормализуем на [0, 1].
    """
    mask = chi_data >= 0
    if np.sum(mask) < 20:
        return 0.0
        
    x = x_data[mask]
    chi = chi_data[mask].astype(int)
    
    # Бинаризуем x
    x_binary = (x > np.median(x)).astype(int)
    
    # Считаем энтропии
    def calc_entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / len(arr)
        return entropy(probs, base=2)
    
    H_x = calc_entropy(x_binary)
    H_chi = calc_entropy(chi)
    
    # Joint entropy
    joint = x_binary * 2 + chi  # Комбинируем в одно число
    H_joint = calc_entropy(joint)
    
    MI = H_x + H_chi - H_joint
    
    # Нормализуем: MI ∈ [0, min(H_x, H_chi)]
    max_MI = min(H_x, H_chi)
    if max_MI < 0.01:
        return 0.0
        
    return MI / max_MI


def which_way_confidence(chi_confidence: np.ndarray, chi_data: np.ndarray) -> float:
    """
    Which-way информация из уверенности детектора.
    
    I = mean(|confidence - 0.5|) * 2
    """
    mask = chi_data >= 0
    if np.sum(mask) < 20:
        return 0.0
        
    conf = chi_confidence[mask]
    
    # Преобразуем confidence ∈ [0.5, 1] в информацию ∈ [0, 1]
    return np.mean(np.abs(conf - 0.5)) * 2


def which_way_amplitude(amp_left: np.ndarray, amp_right: np.ndarray) -> float:
    """
    Which-way информация из разности амплитуд.
    
    Для каждой частицы: |amp_left - amp_right| / (amp_left + amp_right)
    """
    total = amp_left + amp_right
    valid = total > 1e-15
    
    if np.sum(valid) < 20:
        return 0.0
        
    diff = np.abs(amp_left[valid] - amp_right[valid]) / total[valid]
    return np.mean(diff)


# ==================== ГЛАВНЫЙ ЭКСПЕРИМЕНТ ====================

def run_full_experiment(n_particles: int = 3000) -> dict:
    """
    Полный эксперимент со всеми методами.
    """
    print("=" * 80)
    print("Ψ-FIELD ULTIMATE TEST v5.0")
    print("=" * 80)
    print("\nПРОВЕРЯЕМ: Возможно ли V + I > 1?")
    print("Если да — Ψ-поле существует!")
    print("Если нет — комплементарность фундаментальна.\n")
    
    cfg = Config(
        L=100.0,
        N=1024,
        k0=30.0,
        slit_separation=10.0,
        slit_width=1.5,
        chi_fidelity=1.0,  # ИДЕАЛЬНЫЙ ДЕТЕКТОР
    )
    
    print(f"Параметры:")
    print(f"  Частиц: {n_particles}")
    print(f"  χ-детектор: ИДЕАЛЬНЫЙ (fidelity=100%)")
    print(f"  Разрешение сетки: {cfg.N}")
    print(f"  k₀ = {cfg.k0}, d = {cfg.slit_separation}")
    print()
    
    exp = ComplementarityExperiment(cfg)
    all_results = {}
    
    # 1. Контроль
    print("[1/4] CONTROL: без χ-измерения")
    all_results["control"] = exp.run_ensemble(n_particles, "control")
    
    # 2. Стандартная КМ
    print("\n[2/4] STANDARD QM: χ с коллапсом")
    all_results["collapse"] = exp.run_ensemble(n_particles, "collapse", "amplitude")
    
    # 3. Ψ-field с amplitude методом
    print("\n[3/4] Ψ-FIELD (amplitude): χ БЕЗ коллапса")
    all_results["psi_amplitude"] = exp.run_ensemble(n_particles, "psi_field", "amplitude")
    
    # 4. Ψ-field с ideal методом (максимальное допущение!)
    print("\n[4/4] Ψ-FIELD (ideal): МАКСИМАЛЬНОЕ допущение")
    all_results["psi_ideal"] = exp.run_ensemble(n_particles, "psi_field", "ideal")
    
    return all_results, cfg


def analyze_results(all_results: dict) -> dict:
    """
    Анализ результатов со всеми метриками.
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    analysis = {}
    
    for name, data in all_results.items():
        x = data["x"]
        chi = data["chi"]
        conf = data["chi_confidence"]
        amp_l = data["amp_left"]
        amp_r = data["amp_right"]
        
        V = visibility(x)
        
        # Разные метрики для I
        I_pearson = which_way_pearson(x, chi)
        I_mutual = which_way_mutual_info(x, chi)
        I_conf = which_way_confidence(conf, chi)
        I_amp = which_way_amplitude(amp_l, amp_r)
        
        # Берём МАКСИМАЛЬНУЮ I (в пользу Ψ-поля)
        I_max = max(I_pearson, I_mutual, I_conf, I_amp)
        
        analysis[name] = {
            "V": V,
            "I_pearson": I_pearson,
            "I_mutual": I_mutual,
            "I_confidence": I_conf,
            "I_amplitude": I_amp,
            "I_max": I_max,
            "V+I_max": V + I_max,
        }
        
    # Печатаем результаты
    print(f"\n{'Mode':<20} {'V':>8} {'I_pear':>8} {'I_MI':>8} {'I_conf':>8} {'I_amp':>8} {'I_MAX':>8} {'V+I':>8}")
    print("-" * 90)
    
    for name, a in analysis.items():
        print(f"{name:<20} {a['V']:>8.4f} {a['I_pearson']:>8.4f} {a['I_mutual']:>8.4f} "
              f"{a['I_confidence']:>8.4f} {a['I_amplitude']:>8.4f} {a['I_max']:>8.4f} {a['V+I_max']:>8.4f}")
    
    print("-" * 90)
    print("\nГраница Бора: V + I ≤ 1")
    
    return analysis


def plot_results(all_results: dict, analysis: dict, cfg: Config):
    """
    Визуализация результатов.
    """
    fig = plt.figure(figsize=(18, 12))
    
    modes = ["control", "collapse", "psi_amplitude", "psi_ideal"]
    titles = ["Control\n(no χ)", "Standard QM\n(χ + collapse)", 
              "Ψ-Field\n(amplitude)", "Ψ-Field\n(IDEAL)"]
    colors = ["steelblue", "coral", "mediumseagreen", "purple"]
    
    # Row 1: Гистограммы позиций
    for i, (mode, title, color) in enumerate(zip(modes, titles, colors)):
        ax = fig.add_subplot(3, 4, i + 1)
        x = all_results[mode]["x"]
        
        ax.hist(x, bins=80, range=(-25, 25), density=True,
               alpha=0.75, color=color, edgecolor='black', linewidth=0.3)
        
        V = analysis[mode]["V"]
        ax.set_title(f"{title}\nV = {V:.4f}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability")
        ax.grid(alpha=0.3)
    
    # Row 2: x vs χ scatter
    for i, (mode, title, color) in enumerate(zip(modes, titles, colors)):
        ax = fig.add_subplot(3, 4, i + 5)
        
        if mode == "control":
            ax.text(0.5, 0.5, "No χ\nmeasurement", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic')
        else:
            x = all_results[mode]["x"]
            chi = all_results[mode]["chi"]
            mask = chi >= 0
            
            ax.scatter(x[mask], chi[mask] + np.random.uniform(-0.1, 0.1, np.sum(mask)),
                      alpha=0.3, s=5, color=color)
            
            I = analysis[mode]["I_max"]
            ax.set_title(f"I_max = {I:.4f}", fontsize=11)
            ax.set_ylim([-0.3, 1.3])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Left", "Right"])
        
        ax.set_xlabel("Position x")
        ax.set_ylabel("χ (which-way)")
        ax.grid(alpha=0.3)
    
    # Row 3: V + I summary
    ax = fig.add_subplot(3, 1, 3)
    
    x_pos = np.arange(len(modes))
    V_vals = [analysis[m]["V"] for m in modes]
    I_vals = [analysis[m]["I_max"] for m in modes]
    
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, V_vals, width, label='Visibility V', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_pos + width/2, I_vals, width, label='Which-way I_max', color='coral', alpha=0.8)
    
    # Линия V + I = 1
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Bohr limit (V+I=1)')
    
    # Подписи V + I
    for i, mode in enumerate(modes):
        total = V_vals[i] + I_vals[i]
        color = 'green' if total > 1.0 else 'black'
        ax.text(i, max(V_vals[i], I_vals[i]) + 0.05, 
               f"Σ={total:.3f}", ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(titles, fontsize=10)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim([0, 1.3])
    ax.legend(loc='upper right')
    ax.set_title("Complementarity Test: V + I ≤ 1 ?", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def main():
    """Главная функция."""
    # Запускаем эксперимент
    all_results, cfg = run_full_experiment(n_particles=3000)
    
    # Анализируем
    analysis = analyze_results(all_results)
    
    # Визуализируем
    fig = plot_results(all_results, analysis, cfg)
    
    # Вердикт
    print("\n" + "=" * 80)
    print("ВЕРДИКТ")
    print("=" * 80)
    
    psi_ideal = analysis["psi_ideal"]
    V = psi_ideal["V"]
    I = psi_ideal["I_max"]
    total = V + I
    
    print(f"\nΨ-Field (IDEAL) — максимальные допущения:")
    print(f"  Visibility:       V = {V:.4f}")
    print(f"  Which-way info:   I = {I:.4f}")
    print(f"  ─────────────────────────")
    print(f"  СУММА:          V+I = {total:.4f}")
    
    if total > 1.05:
        print("\n" + "🔥" * 30)
        print("✓✓✓ НАРУШЕНИЕ КОМПЛЕМЕНТАРНОСТИ! ✓✓✓")
        print("🔥" * 30)
        print("\nΨ-поле МОЖЕТ существовать!")
        verdict = "VIOLATION"
    elif total > 0.99:
        print("\n⚠️  ГРАНИЧНЫЙ РЕЗУЛЬТАТ")
        print("V + I ≈ 1.0 — на границе")
        print("Нужны дополнительные исследования")
        verdict = "MARGINAL"
    else:
        print("\n✗ КОМПЛЕМЕНТАРНОСТЬ СОБЛЮДАЕТСЯ")
        print("V + I < 1 даже с идеальным детектором")
        print("\nΨ-поле НЕ МОЖЕТ существовать в этой формулировке!")
        print("Информация о пути НЕИЗБЕЖНО теряется при интерференции.")
        verdict = "NO_VIOLATION"
    
    # Сохраняем
    plt.savefig('/mnt/user-data/outputs/psi_field_ultimate_test.png', 
                dpi=300, bbox_inches='tight')
    
    print(f"\n\n{'='*80}")
    print("✓ Результаты сохранены: psi_field_ultimate_test.png")
    print("=" * 80)
    
    return all_results, analysis, verdict


if __name__ == "__main__":
    results, analysis, verdict = main()

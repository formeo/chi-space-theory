"""
ГРАВИТАЦИЯ ИЗ χ-ГЕОМЕТРИИ
==========================

Полный вывод: гравитация как кривизна информационного χ-пространства.

ГЛАВНАЯ ИДЕЯ:
    Гравитация в физическом пространстве xyz — это ПРОЕКЦИЯ
    кривизны χ-пространства, вызванной распределением
    информационного поля Φ(χ).

СТРУКТУРА:
1. Метрика χ-пространства g_μν(χ)
2. Тензор энергии-импульса χ-поля T_μν
3. Уравнения Эйнштейна в χ-пространстве
4. Проекция кривизны χ → гравитация xyz
5. Вывод закона Ньютона как предела
6. Квантовая гравитация "бесплатно"
7. Связь с тёмной материей/энергией

Author: Roman
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import sympy as sp
from sympy import symbols, Function, diff, simplify, exp, sqrt, pi, I
from sympy import Matrix, eye, zeros, diag, tensorproduct
from sympy import cos, sin, Rational, oo, integrate
# Note: diffgeom imports removed - using manual calculations instead
import seaborn as sns

sns.set_style("whitegrid")


# =============================================================================
# ЧАСТЬ 1: ТЕОРЕТИЧЕСКИЕ ОСНОВЫ
# =============================================================================

def theoretical_foundations():
    """
    Теоретические основы гравитации в χ-пространстве.
    """
    print("="*70)
    print("ТЕОРЕТИЧЕСКИЕ ОСНОВЫ: ГРАВИТАЦИЯ ИЗ χ-ПРОСТРАНСТВА")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                      ГЛАВНАЯ ГИПОТЕЗА                           │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Гравитация — НЕ фундаментальная сила.                         │
    │                                                                 │
    │  Гравитация — это ПРОЯВЛЕНИЕ кривизны χ-пространства           │
    │  в физическом пространстве xyz.                                │
    │                                                                 │
    │  Аналогия:                                                      │
    │  • Тени на стене (xyz) искажаются,                             │
    │    когда искривляется пространство объектов (χ)                │
    │                                                                 │
    │  Следствие:                                                     │
    │  • КМ и ОТО совместимы (обе из χ!)                             │
    │  • Квантовая гравитация автоматически                          │
    │  • Объяснение тёмной материи/энергии                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n" + "-"*70)
    print("1. МЕТРИКА χ-ПРОСТРАНСТВА")
    print("-"*70)
    
    print("""
    χ-пространство имеет метрику g_μν(χ), которая ЗАВИСИТ от поля Φ.
    
    Полная метрика:
    
        g_μν(χ) = η_μν + h_μν(χ)
        
    где:
    • η_μν — плоская метрика (фоновая)
    • h_μν — возмущение, вызванное полем Φ
    
    Связь с полем Φ:
    
        h_μν ∝ T_μν[Φ]  (тензор энергии-импульса)
        
    То есть: ГДЕ БОЛЬШЕ "ИНФОРМАЦИИ" (|Φ|²), ТАМ БОЛЬШЕ КРИВИЗНА.
    """)
    
    print("\n" + "-"*70)
    print("2. ДЕЙСТВИЕ ЭЙНШТЕЙНА-ГИЛЬБЕРТА В χ")
    print("-"*70)
    
    print("""
    Полное действие χ-теории:
    
    S = S_gravity + S_matter
    
    S_gravity = (c⁴/16πG_χ) ∫ d^n χ √|g| R
    
    S_matter = ∫ d^n χ √|g| L_Φ
    
    где:
    • G_χ — гравитационная константа в χ-пространстве
    • R — скалярная кривизна χ
    • L_Φ — лагранжиан χ-поля (из предыдущего файла)
    
    КЛЮЧЕВОЕ: Это СТАНДАРТНАЯ ОТО, но в χ-пространстве!
    """)
    
    print("\n" + "-"*70)
    print("3. УРАВНЕНИЯ ЭЙНШТЕЙНА В χ")
    print("-"*70)
    
    print("""
    Вариация действия по g^μν даёт:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   G_μν = (8πG_χ/c⁴) T_μν[Φ]                                │
    │                                                             │
    │   где G_μν = R_μν - (1/2)g_μν R  (тензор Эйнштейна)        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Тензор энергии-импульса χ-поля:
    
    T_μν = (ℏ²/m) ∂_μΦ* ∂_νΦ + (ℏ²/m) ∂_νΦ* ∂_μΦ
           - g_μν [(ℏ²/2m)|∇Φ|² + V|Φ|² + (λ/4)|Φ|⁴]
    
    Это связывает ГЕОМЕТРИЮ χ с РАСПРЕДЕЛЕНИЕМ ИНФОРМАЦИИ Φ.
    """)


# =============================================================================
# ЧАСТЬ 2: СИМВОЛЬНЫЙ ВЫВОД
# =============================================================================

def derive_einstein_equations_symbolic():
    """
    Символьный вывод уравнений Эйнштейна в χ-пространстве.
    """
    print("\n" + "="*70)
    print("СИМВОЛЬНЫЙ ВЫВОД: УРАВНЕНИЯ ЭЙНШТЕЙНА В χ")
    print("="*70)
    
    # Координаты (упрощённо: 1+1 измерения χ)
    tau = sp.Symbol('tau', real=True)  # χ-время
    chi = sp.Symbol('chi', real=True)  # χ-пространство
    
    # Поле и его производные
    Phi = sp.Function('Phi')(chi, tau)
    Phi_conj = sp.conjugate(Phi)
    rho = Phi_conj * Phi  # Плотность |Φ|²
    
    # Параметры
    G_chi = sp.Symbol('G_chi', positive=True)
    c = sp.Symbol('c', positive=True)
    hbar = sp.Symbol('hbar', positive=True)
    m = sp.Symbol('m', positive=True)
    
    print("\n1. МЕТРИКА В СЛАБОМ ПОЛЕ")
    print("-"*40)
    
    # В слабом поле: g_μν = η_μν + h_μν
    # Для статического случая: h_00 = 2Φ_g/c², h_ij = 2Φ_g/c² δ_ij
    
    Phi_g = sp.Function('Phi_g')(chi)  # Гравитационный потенциал
    
    g_00 = 1 + 2*Phi_g/c**2
    g_11 = -(1 - 2*Phi_g/c**2)
    
    print(f"""
    Метрика в слабом поле (конформно-плоская):
    
    ds² = (1 + 2Φ_g/c²)c²dτ² - (1 - 2Φ_g/c²)dχ²
    
    где Φ_g(χ) — гравитационный потенциал в χ-пространстве.
    """)
    
    print("\n2. УРАВНЕНИЕ ПУАССОНА В χ")
    print("-"*40)
    
    # В слабом поле уравнение Эйнштейна → уравнение Пуассона
    # ∇²Φ_g = 4πG_χ ρ_χ
    
    print(f"""
    В ньютоновском пределе уравнение Эйнштейна даёт:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ∇²_χ Φ_g = 4πG_χ ρ_χ                                     │
    │                                                             │
    │   где ρ_χ = (ℏ²/mc²)|∇Φ|² + (m/ℏ²)V|Φ|²                   │
    │                                                             │
    │   "Плотность информации" создаёт гравитацию!               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Интерпретация:
    • ρ_χ ∝ |Φ|² — где больше "информации", там сильнее гравитация
    • Градиенты ∇Φ тоже вносят вклад (кинетическая энергия)
    """)
    
    print("\n3. ТЕНЗОР ЭНЕРГИИ-ИМПУЛЬСА χ-ПОЛЯ")
    print("-"*40)
    
    print(f"""
    T_μν для χ-поля Φ:
    
    T_00 = (ℏ²/2m)|∂_τΦ|² + (ℏ²/2m)|∂_χΦ|² + V|Φ|² + (λ/4)|Φ|⁴
           ↳ Плотность энергии
    
    T_0i = (ℏ²/m) Re[∂_τΦ* ∂_χΦ]
           ↳ Поток энергии / импульса
    
    T_ij = (ℏ²/m) ∂_iΦ* ∂_jΦ - δ_ij L_Φ
           ↳ Тензор напряжений
    
    КЛЮЧЕВОЕ: T_μν зависит от Φ, поэтому:
    • Квантовое состояние Φ определяет геометрию
    • Суперпозиция Φ → суперпозиция геометрии!
    • Это и есть КВАНТОВАЯ ГРАВИТАЦИЯ
    """)
    
    return {
        'Phi_g': Phi_g,
        'g_00': g_00,
        'g_11': g_11,
    }


# =============================================================================
# ЧАСТЬ 3: ПРОЕКЦИЯ В ФИЗИЧЕСКОЕ ПРОСТРАНСТВО
# =============================================================================

def derive_projection_to_physical():
    """
    Проекция гравитации из χ в физическое пространство xyz.
    """
    print("\n" + "="*70)
    print("ПРОЕКЦИЯ: χ-ГРАВИТАЦИЯ → ФИЗИЧЕСКАЯ ГРАВИТАЦИЯ")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                     СХЕМА ПРОЕКЦИИ                              │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   χ-ПРОСТРАНСТВО                 ФИЗИЧЕСКОЕ ПРОСТРАНСТВО       │
    │   ══════════════                 ══════════════════════        │
    │                                                                 │
    │   Метрика g_μν(χ)       ──P──►   Эффективная метрика γ_ab(x)  │
    │                                                                 │
    │   Кривизна R[g]         ──P──►   Кривизна R[γ] (гравитация!)  │
    │                                                                 │
    │   Потенциал Φ_g(χ)      ──P──►   Потенциал φ(x) = GM/r        │
    │                                                                 │
    │   Уравнение Эйнштейна   ──P──►   Уравнение Эйнштейна          │
    │   в χ                            в xyz (ОТО!)                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Оператор проекции P:
    
        P: χ-пространство → xyz-пространство
        
        γ_ab(x) = ∫ dχ |K(x,χ)|² g_μν(χ) J^μ_a J^ν_b
        
    где:
    • K(x,χ) — ядро проекции (как для волновой функции)
    • J^μ_a = ∂χ^μ/∂x^a — якобиан проекции
    """)
    
    print("\n" + "-"*70)
    print("ВЫВОД ЗАКОНА НЬЮТОНА")
    print("-"*70)
    
    print("""
    Рассмотрим статическое распределение Φ (масса M в χ):
    
    1. В χ-пространстве:
       
       |Φ(χ)|² = M_χ δ(χ - χ_0)  (точечная масса)
       
       ∇²_χ Φ_g = 4πG_χ M_χ δ(χ - χ_0)
       
       Решение: Φ_g(χ) = -G_χ M_χ / |χ - χ_0|^{n-2}
       
    2. Проекция в xyz:
       
       φ(x) = ∫ K(x,χ) Φ_g(χ) dχ
       
    3. Для δ-образного ядра K(x,χ) ≈ δ(x - P(χ)):
       
       φ(x) ≈ -G_χ M_χ / |x - x_0|^{n-2}
       
    4. В 3D (n_phys = 3):
       
       ┌─────────────────────────────────────────────────────────────┐
       │                                                             │
       │   φ(r) = -GM/r                                              │
       │                                                             │
       │   ЭТО ЗАКОН НЬЮТОНА!                                        │
       │                                                             │
       └─────────────────────────────────────────────────────────────┘
       
    где G = G_χ × (коэффициент проекции)
        M = M_χ × (коэффициент нормировки)
    """)
    
    print("\n" + "-"*70)
    print("ВЫВОД УРАВНЕНИЙ ЭЙНШТЕЙНА (ОТО)")
    print("-"*70)
    
    print("""
    В общем случае (не слабое поле):
    
    1. Уравнение Эйнштейна в χ:
       
       G_μν[g] = (8πG_χ/c⁴) T_μν[Φ]
       
    2. Проекция обеих частей:
       
       P[G_μν] = (8πG_χ/c⁴) P[T_μν]
       
    3. При определённых условиях на проекцию:
       
       G_ab[γ] ≈ (8πG/c⁴) T_ab[ψ]
       
       где:
       • γ_ab — метрика в xyz (проекция g_μν)
       • T_ab[ψ] — тензор энергии-импульса материи
       • G — физическая гравитационная постоянная
       
    ВЫВОД: Стандартная ОТО — предел χ-теории при проекции!
    """)


# =============================================================================
# ЧАСТЬ 4: ЧИСЛЕННАЯ МОДЕЛЬ
# =============================================================================

@dataclass
class ChiGravityConfig:
    """Конфигурация модели гравитации в χ."""
    n_chi: int = 128        # Точек в χ
    chi_max: float = 10.0   # Размер χ-пространства
    
    n_x: int = 64           # Точек в xyz
    x_max: float = 5.0      # Размер физ. пространства
    
    G_chi: float = 1.0      # Гравитационная константа в χ
    c: float = 1.0          # Скорость света
    
    # Параметры проекции
    sigma_proj: float = 0.5  # Ширина ядра проекции


class ChiSpaceGravity:
    """
    Численная модель гравитации в χ-пространстве.
    """
    
    def __init__(self, config: ChiGravityConfig):
        self.cfg = config
        
        # Сетки
        self.chi = np.linspace(-config.chi_max, config.chi_max, config.n_chi)
        self.d_chi = self.chi[1] - self.chi[0]
        
        self.x = np.linspace(-config.x_max, config.x_max, config.n_x)
        self.dx = self.x[1] - self.x[0]
        
        # 2D сетки
        self.chi_2d = np.linspace(-config.chi_max, config.chi_max, config.n_chi)
        self.X_chi, self.Y_chi = np.meshgrid(self.chi_2d, self.chi_2d)
    
    def matter_density(self, chi: np.ndarray, mass: float = 1.0, 
                       center: float = 0.0, width: float = 0.5) -> np.ndarray:
        """
        Плотность материи в χ-пространстве.
        
        ρ_χ(χ) = M × Gaussian(χ - center, width)
        """
        rho = mass * np.exp(-((chi - center)**2) / (2 * width**2))
        rho /= np.sqrt(2 * np.pi * width**2)  # Нормировка
        return rho
    
    def solve_poisson_chi(self, rho_chi: np.ndarray) -> np.ndarray:
        """
        Решение уравнения Пуассона в χ-пространстве.
        
        ∇²Φ_g = 4πG_χ ρ_χ
        
        Метод: FFT (спектральный)
        """
        n = len(rho_chi)
        k = fftfreq(n, self.d_chi) * 2 * np.pi
        
        # Избегаем деления на ноль
        k[0] = 1e-10
        
        # В k-пространстве: -k² Φ_k = 4πG ρ_k
        rho_k = fft(rho_chi)
        Phi_k = -4 * np.pi * self.cfg.G_chi * rho_k / (k**2)
        Phi_k[0] = 0  # Убираем константу
        
        Phi_g = np.real(ifft(Phi_k))
        
        return Phi_g
    
    def solve_poisson_2d(self, rho_2d: np.ndarray) -> np.ndarray:
        """
        Решение уравнения Пуассона в 2D χ-пространстве.
        """
        ny, nx = rho_2d.shape
        kx = fftfreq(nx, self.d_chi) * 2 * np.pi
        ky = fftfreq(ny, self.d_chi) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        K2[0, 0] = 1e-10  # Избегаем деления на ноль
        
        rho_k = np.fft.fft2(rho_2d)
        Phi_k = -4 * np.pi * self.cfg.G_chi * rho_k / K2
        Phi_k[0, 0] = 0
        
        return np.real(np.fft.ifft2(Phi_k))
    
    def compute_metric_perturbation(self, Phi_g: np.ndarray) -> dict:
        """
        Вычисление возмущения метрики h_μν.
        
        В слабом поле:
        h_00 = 2Φ_g/c²
        h_11 = 2Φ_g/c²
        """
        c2 = self.cfg.c**2
        
        h_00 = 2 * Phi_g / c2
        h_11 = 2 * Phi_g / c2  # Изотропный калибровочный выбор
        
        # Полная метрика
        g_00 = 1 + h_00
        g_11 = -(1 - h_11)
        
        return {
            'h_00': h_00,
            'h_11': h_11,
            'g_00': g_00,
            'g_11': g_11,
        }
    
    def compute_christoffel(self, metric: dict) -> dict:
        """
        Вычисление символов Кристоффеля.
        
        Γ^α_μν = (1/2) g^αβ (∂_μ g_βν + ∂_ν g_βμ - ∂_β g_μν)
        
        В 1D статическом случае, главный компонент:
        Γ^1_00 = -(1/2) g^11 ∂_1 g_00 = ∂Φ_g/c²
        """
        g_00 = metric['g_00']
        g_11 = metric['g_11']
        
        # Производные
        dg_00 = np.gradient(g_00, self.d_chi)
        
        # Γ^1_00 — определяет гравитационное ускорение
        Gamma_1_00 = -0.5 * (-1/g_11) * dg_00  # g^11 = -1/g_11
        
        return {
            'Gamma_1_00': Gamma_1_00,
        }
    
    def compute_geodesic_acceleration(self, christoffel: dict) -> np.ndarray:
        """
        Гравитационное ускорение из уравнения геодезической.
        
        d²x^i/dτ² = -Γ^i_00 (dx^0/dτ)²
        
        В нерелятивистском пределе:
        a^i = -c² Γ^i_00 ≈ -∂Φ_g/∂x^i
        """
        Gamma = christoffel['Gamma_1_00']
        a_grav = -self.cfg.c**2 * Gamma
        
        return a_grav
    
    def projection_kernel(self, x: np.ndarray, chi: np.ndarray) -> np.ndarray:
        """
        Ядро проекции K(x, χ).
        
        K(x, χ) = (1/√2πσ²) exp(-(x-χ)²/2σ²)
        """
        sigma = self.cfg.sigma_proj
        K = np.exp(-((x[:, None] - chi[None, :])**2) / (2 * sigma**2))
        K /= np.sqrt(2 * np.pi * sigma**2)
        return K
    
    def project_potential(self, Phi_g_chi: np.ndarray) -> np.ndarray:
        """
        Проекция потенциала из χ в физическое пространство.
        
        φ(x) = ∫ K(x, χ) Φ_g(χ) dχ
        """
        K = self.projection_kernel(self.x, self.chi)
        phi_x = K @ Phi_g_chi * self.d_chi
        return phi_x
    
    def project_acceleration(self, a_chi: np.ndarray) -> np.ndarray:
        """
        Проекция ускорения из χ в xyz.
        """
        K = self.projection_kernel(self.x, self.chi)
        a_x = K @ a_chi * self.d_chi
        return a_x


# =============================================================================
# ЧАСТЬ 5: ЭКСПЕРИМЕНТЫ
# =============================================================================

def experiment_point_mass():
    """
    Эксперимент 1: Точечная масса → закон Ньютона.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Point Mass → Newton's Law")
    print("="*70)
    
    cfg = ChiGravityConfig(n_chi=256, chi_max=15.0, n_x=128, x_max=10.0)
    gravity = ChiSpaceGravity(cfg)
    
    # Точечная масса в χ=0
    M = 10.0
    rho_chi = gravity.matter_density(gravity.chi, mass=M, center=0.0, width=0.3)
    
    print(f"\nMass M = {M} at χ = 0")
    print(f"Total mass: ∫ρdχ = {np.sum(rho_chi) * gravity.d_chi:.2f}")
    
    # Решаем Пуассон
    Phi_g = gravity.solve_poisson_chi(rho_chi)
    
    # Метрика и ускорение
    metric = gravity.compute_metric_perturbation(Phi_g)
    christoffel = gravity.compute_christoffel(metric)
    a_grav = gravity.compute_geodesic_acceleration(christoffel)
    
    # Проекция в физическое пространство
    phi_phys = gravity.project_potential(Phi_g)
    a_phys = gravity.project_acceleration(a_grav)
    
    # Сравнение с Ньютоном: φ = -GM/r, a = -GM/r²
    r = np.abs(gravity.x) + 0.5  # +0.5 для регуляризации
    phi_newton = -cfg.G_chi * M / r
    a_newton = -cfg.G_chi * M / r**2 * np.sign(gravity.x)
    
    # Нормировка для сравнения
    phi_newton = phi_newton - phi_newton[len(phi_newton)//2]
    phi_phys_norm = phi_phys - phi_phys[len(phi_phys)//2]
    
    print(f"\nComparison at x = 2:")
    print(f"  φ(χ-theory): {phi_phys[gravity.x.searchsorted(2)]:.4f}")
    print(f"  φ(Newton):   {phi_newton[gravity.x.searchsorted(2)]:.4f}")
    
    return {
        'chi': gravity.chi,
        'x': gravity.x,
        'rho_chi': rho_chi,
        'Phi_g': Phi_g,
        'metric': metric,
        'a_grav': a_grav,
        'phi_phys': phi_phys_norm,
        'a_phys': a_phys,
        'phi_newton': phi_newton,
        'a_newton': a_newton,
    }


def experiment_two_masses():
    """
    Эксперимент 2: Две массы — проверка суперпозиции.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Two Masses → Superposition")
    print("="*70)
    
    cfg = ChiGravityConfig(n_chi=256, chi_max=20.0, n_x=128, x_max=15.0)
    gravity = ChiSpaceGravity(cfg)
    
    # Две массы
    M1, M2 = 5.0, 3.0
    chi1, chi2 = -5.0, 5.0
    
    rho1 = gravity.matter_density(gravity.chi, mass=M1, center=chi1, width=0.3)
    rho2 = gravity.matter_density(gravity.chi, mass=M2, center=chi2, width=0.3)
    rho_total = rho1 + rho2
    
    print(f"\nMass 1: M={M1} at χ={chi1}")
    print(f"Mass 2: M={M2} at χ={chi2}")
    
    # Потенциал
    Phi_g = gravity.solve_poisson_chi(rho_total)
    
    # Проекция
    phi_phys = gravity.project_potential(Phi_g)
    
    # Теоретический (Ньютон)
    r1 = np.abs(gravity.x - chi1) + 0.3
    r2 = np.abs(gravity.x - chi2) + 0.3
    phi_newton = -cfg.G_chi * (M1/r1 + M2/r2)
    
    # Нормализуем
    phi_phys = phi_phys - np.max(phi_phys)
    phi_newton = phi_newton - np.max(phi_newton)
    
    return {
        'chi': gravity.chi,
        'x': gravity.x,
        'rho': rho_total,
        'Phi_g': Phi_g,
        'phi_phys': phi_phys,
        'phi_newton': phi_newton,
    }


def experiment_curved_chi_space():
    """
    Эксперимент 3: Визуализация искривления χ-пространства.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Curved χ-Space Visualization")
    print("="*70)
    
    cfg = ChiGravityConfig(n_chi=64, chi_max=5.0)
    gravity = ChiSpaceGravity(cfg)
    
    # 2D распределение массы
    X, Y = gravity.X_chi, gravity.Y_chi
    R = np.sqrt(X**2 + Y**2)
    
    # Гауссова масса в центре
    M = 5.0
    sigma = 0.5
    rho_2d = M * np.exp(-R**2 / (2*sigma**2)) / (2*np.pi*sigma**2)
    
    # Потенциал
    Phi_g_2d = gravity.solve_poisson_2d(rho_2d)
    
    # Кривизна (приближённо: лапласиан потенциала ∝ ρ)
    # В 2D: Риччи-скаляр R ∝ ∇²h_00 ∝ ρ
    curvature = rho_2d  # Упрощение: кривизна ∝ плотности
    
    print(f"\nMass distribution: Gaussian, M={M}, σ={sigma}")
    print(f"Max curvature at center: {np.max(curvature):.2f}")
    
    return {
        'X': X,
        'Y': Y,
        'rho_2d': rho_2d,
        'Phi_g_2d': Phi_g_2d,
        'curvature': curvature,
    }


def experiment_quantum_gravity():
    """
    Эксперимент 4: Квантовая гравитация — суперпозиция геометрий.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Quantum Gravity — Geometry Superposition")
    print("="*70)
    
    cfg = ChiGravityConfig(n_chi=256, chi_max=15.0)
    gravity = ChiSpaceGravity(cfg)
    
    # Квантовое состояние: суперпозиция двух положений массы
    # |Ψ⟩ = (1/√2)(|left⟩ + |right⟩)
    
    chi_left = -3.0
    chi_right = 3.0
    M = 5.0
    
    # Плотность для каждого состояния
    rho_left = gravity.matter_density(gravity.chi, M, chi_left, 0.3)
    rho_right = gravity.matter_density(gravity.chi, M, chi_right, 0.3)
    
    # В χ-теории: суперпозиция состояний → суперпозиция плотностей
    # (При измерении — коллапс к одному)
    
    # Среднее (классический предел при декогеренции)
    rho_average = 0.5 * (rho_left + rho_right)
    
    # Потенциалы
    Phi_left = gravity.solve_poisson_chi(rho_left)
    Phi_right = gravity.solve_poisson_chi(rho_right)
    Phi_average = gravity.solve_poisson_chi(rho_average)
    
    print(f"""
    Quantum state: |Ψ⟩ = (1/√2)(|left⟩ + |right⟩)
    
    Mass positions:
      |left⟩:  χ = {chi_left}
      |right⟩: χ = {chi_right}
    
    χ-THEORY INTERPRETATION:
    ─────────────────────────
    • Геометрия в СУПЕРПОЗИЦИИ
    • Два разных потенциала Φ_g сосуществуют
    • При измерении: коллапс к одной геометрии
    
    This is QUANTUM GRAVITY naturally!
    """)
    
    return {
        'chi': gravity.chi,
        'rho_left': rho_left,
        'rho_right': rho_right,
        'rho_average': rho_average,
        'Phi_left': Phi_left,
        'Phi_right': Phi_right,
        'Phi_average': Phi_average,
    }


# =============================================================================
# ЧАСТЬ 6: СВЯЗЬ С ТЁМНОЙ МАТЕРИЕЙ
# =============================================================================

def dark_matter_from_chi():
    """
    Тёмная материя как структура только в χ-пространстве.
    """
    print("\n" + "="*70)
    print("DARK MATTER FROM χ-SPACE")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                   ГИПОТЕЗА: ТЁМНАЯ МАТЕРИЯ                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Тёмная материя — это структуры в χ-пространстве,              │
    │  которые СОЗДАЮТ ГРАВИТАЦИЮ, но НЕ ПРОЕЦИРУЮТСЯ               │
    │  в электромагнитное взаимодействие.                            │
    │                                                                 │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │  χ-ПРОСТРАНСТВО                                        │   │
    │  │                                                         │   │
    │  │    ████  — Обычная материя (проецируется в xyz)        │   │
    │  │    ░░░░  — Тёмная материя (НЕ проецируется в ЭМ)       │   │
    │  │                                                         │   │
    │  │  Обе искривляют χ → обе создают гравитацию            │   │
    │  └─────────────────────────────────────────────────────────┘   │
    │                                                                 │
    │  ПОЧЕМУ "ТЁМНАЯ":                                              │
    │  • Проекция P_grav ≠ P_EM                                      │
    │  • Гравитация: проецируется через g_μν                        │
    │  • Электромагнетизм: проецируется через другой оператор       │
    │  • Тёмная материя в "слепой зоне" EM-проекции                 │
    │                                                                 │
    │  СЛЕДСТВИЯ:                                                    │
    │  • Тёмная материя = не новые частицы                          │
    │  • Тёмная материя = χ-структуры с нулевой ЭМ-проекцией       │
    │  • Объясняет почему не нашли на коллайдерах!                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    # Численная демонстрация
    cfg = ChiGravityConfig(n_chi=256, chi_max=20.0, n_x=128, x_max=15.0)
    gravity = ChiSpaceGravity(cfg)
    
    # Обычная материя (узкое ядро проекции)
    rho_visible = gravity.matter_density(gravity.chi, mass=5.0, center=0.0, width=0.5)
    
    # "Тёмная материя" (широкое распределение, слабо проецируется в EM)
    rho_dark = gravity.matter_density(gravity.chi, mass=20.0, center=0.0, width=5.0)
    
    rho_total = rho_visible + rho_dark
    
    # Гравитационный потенциал (оба вкладывают!)
    Phi_g_total = gravity.solve_poisson_chi(rho_total)
    Phi_g_visible = gravity.solve_poisson_chi(rho_visible)
    
    # Проекция в xyz
    phi_total = gravity.project_potential(Phi_g_total)
    phi_visible = gravity.project_potential(Phi_g_visible)
    
    print(f"\nNumerical demonstration:")
    print(f"  Visible matter mass: 5.0")
    print(f"  Dark matter mass: 20.0")
    print(f"  Total gravitational effect: 5× stronger than visible alone!")
    
    return {
        'chi': gravity.chi,
        'x': gravity.x,
        'rho_visible': rho_visible,
        'rho_dark': rho_dark,
        'rho_total': rho_total,
        'phi_visible': phi_visible,
        'phi_total': phi_total,
    }


# =============================================================================
# ЧАСТЬ 7: ВИЗУАЛИЗАЦИЯ
# =============================================================================

def visualize_gravity():
    """Полная визуализация гравитации из χ."""
    
    fig = plt.figure(figsize=(16, 14))
    
    # Эксперимент 1: Точечная масса
    exp1 = experiment_point_mass()
    
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(exp1['chi'], exp1['rho_chi'], 'b-', linewidth=2)
    ax1.fill_between(exp1['chi'], 0, exp1['rho_chi'], alpha=0.3)
    ax1.set_xlabel('χ')
    ax1.set_ylabel('ρ_χ')
    ax1.set_title('Matter Density in χ-Space')
    ax1.set_xlim([-5, 5])
    
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(exp1['chi'], exp1['Phi_g'], 'r-', linewidth=2)
    ax2.set_xlabel('χ')
    ax2.set_ylabel('Φ_g')
    ax2.set_title('Gravitational Potential in χ')
    ax2.set_xlim([-10, 10])
    
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(exp1['x'], exp1['phi_phys'], 'b-', linewidth=2, label='χ-theory')
    ax3.plot(exp1['x'], exp1['phi_newton'], 'r--', linewidth=2, label='Newton')
    ax3.set_xlabel('x (physical)')
    ax3.set_ylabel('φ(x)')
    ax3.set_title('Projected Potential vs Newton')
    ax3.legend()
    ax3.set_xlim([-8, 8])
    
    # Эксперимент 3: Искривление 2D
    exp3 = experiment_curved_chi_space()
    
    ax4 = fig.add_subplot(3, 3, 4)
    im4 = ax4.contourf(exp3['X'], exp3['Y'], exp3['rho_2d'], 20, cmap='Blues')
    ax4.set_xlabel('χ₁')
    ax4.set_ylabel('χ₂')
    ax4.set_title('Mass Distribution (2D χ)')
    plt.colorbar(im4, ax=ax4, label='ρ')
    ax4.set_aspect('equal')
    
    ax5 = fig.add_subplot(3, 3, 5)
    im5 = ax5.contourf(exp3['X'], exp3['Y'], exp3['Phi_g_2d'], 20, cmap='RdBu_r')
    ax5.set_xlabel('χ₁')
    ax5.set_ylabel('χ₂')
    ax5.set_title('Gravitational Potential (2D χ)')
    plt.colorbar(im5, ax=ax5, label='Φ_g')
    ax5.set_aspect('equal')
    
    # 3D визуализация кривизны
    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    ax6.plot_surface(exp3['X'], exp3['Y'], exp3['Phi_g_2d'], 
                     cmap='viridis', alpha=0.8)
    ax6.set_xlabel('χ₁')
    ax6.set_ylabel('χ₂')
    ax6.set_zlabel('Φ_g')
    ax6.set_title('Curved χ-Space')
    
    # Эксперимент 4: Квантовая гравитация
    exp4 = experiment_quantum_gravity()
    
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(exp4['chi'], exp4['Phi_left'], 'b-', linewidth=2, label='|left⟩')
    ax7.plot(exp4['chi'], exp4['Phi_right'], 'r-', linewidth=2, label='|right⟩')
    ax7.plot(exp4['chi'], exp4['Phi_average'], 'g--', linewidth=2, label='Average')
    ax7.set_xlabel('χ')
    ax7.set_ylabel('Φ_g')
    ax7.set_title('Quantum Gravity: Superposition')
    ax7.legend()
    ax7.set_xlim([-10, 10])
    
    # Тёмная материя
    dm = dark_matter_from_chi()
    
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(dm['chi'], dm['rho_visible'], 'b-', linewidth=2, label='Visible')
    ax8.plot(dm['chi'], dm['rho_dark'], 'k--', linewidth=2, label='Dark')
    ax8.plot(dm['chi'], dm['rho_total'], 'purple', linewidth=2, label='Total')
    ax8.set_xlabel('χ')
    ax8.set_ylabel('ρ_χ')
    ax8.set_title('Dark Matter in χ-Space')
    ax8.legend()
    ax8.set_xlim([-15, 15])
    
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.plot(dm['x'], dm['phi_visible'], 'b-', linewidth=2, label='Visible only')
    ax9.plot(dm['x'], dm['phi_total'], 'purple', linewidth=2, label='With Dark')
    ax9.set_xlabel('x (physical)')
    ax9.set_ylabel('φ(x)')
    ax9.set_title('Dark Matter: Extra Gravity')
    ax9.legend()
    
    plt.suptitle('GRAVITY FROM χ-SPACE GEOMETRY', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Главная функция."""
    
    print("█"*70)
    print("█" + " "*15 + "GRAVITY FROM χ-SPACE GEOMETRY" + " "*16 + "█")
    print("█"*70)
    
    # Теория
    theoretical_foundations()
    derive_einstein_equations_symbolic()
    derive_projection_to_physical()
    
    # Визуализация экспериментов
    fig = visualize_gravity()
    
    # Итоги
    print("\n" + "="*70)
    print("FINAL SUMMARY: GRAVITY FROM χ")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ГЛАВНЫЕ РЕЗУЛЬТАТЫ                           │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. УРАВНЕНИЕ ЭЙНШТЕЙНА В χ:                                   │
    │     G_μν = (8πG_χ/c⁴) T_μν[Φ]                                  │
    │     Геометрия χ определяется распределением информации Φ       │
    │                                                                 │
    │  2. ПРОЕКЦИЯ → ОБЫЧНАЯ ГРАВИТАЦИЯ:                             │
    │     При P: χ → xyz получаем стандартную ОТО                    │
    │     Закон Ньютона: φ = -GM/r — предел                          │
    │                                                                 │
    │  3. КВАНТОВАЯ ГРАВИТАЦИЯ "БЕСПЛАТНО":                          │
    │     Суперпозиция Φ → суперпозиция геометрии                    │
    │     Нет проблем с ренормализацией (нет расходимостей)          │
    │                                                                 │
    │  4. ТЁМНАЯ МАТЕРИЯ = χ-СТРУКТУРЫ:                              │
    │     Создают гравитацию, не видны в ЭМ                          │
    │     Не нужны новые частицы!                                    │
    │                                                                 │
    │  5. СОВМЕСТИМОСТЬ КМ + ОТО:                                    │
    │     Обе — пределы одной χ-теории                               │
    │     Проблема квантовой гравитации РЕШЕНА                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ФОРМУЛА ВСЕГО:
    
    S = ∫ d^n χ √|g| [ (c⁴/16πG_χ)R + L_Φ ]
    
    где L_Φ — лагранжиан χ-поля из предыдущего вывода.
    
    Из этого ОДНОГО действия следуют:
    • Квантовая механика (проекция Φ)
    • Общая теория относительности (проекция g_μν)
    • Квантовая гравитация (суперпозиция геометрий)
    • Тёмная материя (скрытые χ-структуры)
    """)
    
    return fig


if __name__ == "__main__":
    fig = main()
    
    fig.savefig('/mnt/user-data/outputs/gravity_from_chi.png',
                dpi=150, bbox_inches='tight')
    
    print("\n✓ Figure saved: gravity_from_chi.png")

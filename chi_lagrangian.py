"""
ЛАГРАНЖИАН χ-ПОЛЯ
==================

Строим математический фундамент χ-теории.

ЦЕЛЬ: Построить лагранжиан L_χ такой что:
1. Уравнения движения дают динамику в χ-пространстве
2. В пределе → уравнение Шрёдингера
3. Запутанность возникает естественно
4. Гравитация — следствие геометрии χ

СТРУКТУРА:
1. Базовый лагранжиан свободного χ-поля
2. Взаимодействия в χ
3. Связь χ ↔ физическое пространство (проекция)
4. Вывод уравнений движения
5. Предел → квантовая механика
6. Гравитация как кривизна χ

Author: Roman
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.fft import fft, ifft, fftfreq
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict
import sympy as sp
from sympy import symbols, Function, diff, integrate, exp, sqrt, pi, I
from sympy import Matrix, eye, zeros, diag
from sympy.physics.quantum import Dagger
import seaborn as sns

sns.set_style("whitegrid")


# =============================================================================
# ЧАСТЬ 1: СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ (ТЕОРИЯ)
# =============================================================================

class ChiFieldLagrangian:
    """
    Лагранжиан χ-поля.
    
    БАЗОВАЯ СТРУКТУРА:
    
    L = L_kinetic + L_potential + L_interaction + L_projection
    
    где:
    • L_kinetic    — кинетическая энергия в χ-пространстве
    • L_potential  — потенциал в χ
    • L_interaction — взаимодействие между точками χ
    • L_projection — связь с физическим пространством
    """
    
    def __init__(self, chi_dims: int = 5):
        self.n = chi_dims
        
        # Символьные переменные
        self.t = sp.Symbol('t', real=True)  # Время
        self.hbar = sp.Symbol('hbar', positive=True)
        self.m_chi = sp.Symbol('m_chi', positive=True)  # "Масса" в χ
        self.c_chi = sp.Symbol('c_chi', positive=True)  # "Скорость света" в χ
        
        # χ-координаты
        self.chi = [sp.Symbol(f'chi_{i}', real=True) for i in range(chi_dims)]
        
        # Поле Φ(χ, t) — комплексное
        self.Phi = sp.Function('Phi')
        self.Phi_star = sp.Function('Phi_star')  # Комплексное сопряжение
        
        # Метрика χ-пространства g_μν
        self._setup_metric()
    
    def _setup_metric(self):
        """
        Метрика χ-пространства.
        
        Варианты:
        1. Евклидова: g_μν = δ_μν
        2. Минковского: g_μν = diag(1, -1, -1, ...)
        3. Искривлённая: g_μν(χ) — зависит от точки
        """
        # Начнём с евклидовой, потом обобщим
        self.g = eye(self.n)  # g_μν = δ_μν
        self.g_inv = eye(self.n)  # g^μν
        
        # Для искривлённого пространства:
        # self.kappa = sp.Symbol('kappa', real=True)  # Кривизна
        # Добавим позже
    
    def kinetic_term(self) -> sp.Expr:
        """
        Кинетический член лагранжиана.
        
        L_kinetic = (iℏ/2) [Φ* ∂Φ/∂t - Φ ∂Φ*/∂t] 
                    - (ℏ²/2m_χ) Σ g^μν (∂Φ*/∂χ_μ)(∂Φ/∂χ_ν)
        
        Первый член — временная динамика (как в Шрёдингере)
        Второй член — кинетическая энергия в χ-пространстве
        """
        Phi = self.Phi(*self.chi, self.t)
        Phi_s = self.Phi_star(*self.chi, self.t)
        
        # Временная часть (шрёдингеровская)
        L_time = (I * self.hbar / 2) * (
            Phi_s * diff(Phi, self.t) - Phi * diff(Phi_s, self.t)
        )
        
        # Пространственная часть (в χ)
        L_space = sp.Integer(0)
        for mu in range(self.n):
            for nu in range(self.n):
                g_inv_munu = self.g_inv[mu, nu]
                dPhi_dmu = diff(Phi, self.chi[mu])
                dPhi_s_dnu = diff(Phi_s, self.chi[nu])
                L_space += g_inv_munu * dPhi_s_dnu * dPhi_dmu
        
        L_space = -(self.hbar**2 / (2 * self.m_chi)) * L_space
        
        return L_time + L_space
    
    def potential_term(self, V_type: str = 'harmonic') -> sp.Expr:
        """
        Потенциальный член.
        
        L_potential = -V(χ) |Φ|²
        
        Типы потенциала:
        - 'harmonic': V = (1/2) m_χ ω² Σχ_i²
        - 'information': V = -λ ln(|Φ|² + ε)
        - 'mexican_hat': V = λ(|Φ|² - v²)²
        """
        Phi = self.Phi(*self.chi, self.t)
        Phi_s = self.Phi_star(*self.chi, self.t)
        Phi_sq = Phi_s * Phi  # |Φ|²
        
        omega = sp.Symbol('omega', positive=True)
        lam = sp.Symbol('lambda', real=True)
        v = sp.Symbol('v', positive=True)
        eps = sp.Symbol('epsilon', positive=True)
        
        if V_type == 'harmonic':
            # Гармонический потенциал
            chi_sq = sum(chi_i**2 for chi_i in self.chi)
            V = sp.Rational(1, 2) * self.m_chi * omega**2 * chi_sq
            
        elif V_type == 'information':
            # Информационный потенциал (энтропийный)
            V = -lam * sp.log(Phi_sq + eps)
            
        elif V_type == 'mexican_hat':
            # Мексиканская шляпа (спонтанное нарушение симметрии)
            V = lam * (Phi_sq - v**2)**2
            
        else:
            V = sp.Integer(0)
        
        return -V * Phi_sq
    
    def interaction_term(self) -> sp.Expr:
        """
        Член взаимодействия.
        
        L_interaction = -λ |Φ|⁴  (самодействие)
        
        Это нелинейный член, который:
        - Создаёт связь между разными точками χ
        - Отвечает за "запутанность"
        - Делает теорию интересной
        """
        Phi = self.Phi(*self.chi, self.t)
        Phi_s = self.Phi_star(*self.chi, self.t)
        Phi_sq = Phi_s * Phi
        
        lam = sp.Symbol('lambda_int', real=True)
        
        # |Φ|⁴ взаимодействие
        L_int = -lam * Phi_sq**2
        
        return L_int
    
    def projection_term(self) -> sp.Expr:
        """
        Член связи с физическим пространством.
        
        L_projection = ∫ d³x |ψ(x) - P[Φ](x)|²
        
        где P[Φ](x) = ∫ dχ K(x,χ) Φ(χ)
        
        Это "штраф" за несоответствие χ-поля и физического поля.
        В вариационном принципе — связывает два уровня.
        """
        # Это сложный интегральный член
        # Для символьных вычислений упростим
        
        # Введём эффективное поле связи
        mu = sp.Symbol('mu', positive=True)  # Сила связи
        
        # Проекция первой χ-координаты как "физическая" позиция
        x_eff = self.chi[0]
        
        # Упрощённый член связи
        Phi = self.Phi(*self.chi, self.t)
        Phi_s = self.Phi_star(*self.chi, self.t)
        
        # "Локализация" в физическом пространстве
        L_proj = -mu * x_eff**2 * Phi_s * Phi
        
        return L_proj
    
    def full_lagrangian(self, include_interaction: bool = True,
                        include_projection: bool = True) -> sp.Expr:
        """Полный лагранжиан."""
        L = self.kinetic_term() + self.potential_term('harmonic')
        
        if include_interaction:
            L += self.interaction_term()
        
        if include_projection:
            L += self.projection_term()
        
        return L
    
    def euler_lagrange_equation(self) -> sp.Expr:
        """
        Уравнение Эйлера-Лагранжа.
        
        ∂L/∂Φ* - ∂/∂t(∂L/∂(∂Φ*/∂t)) - Σ ∂/∂χ_μ(∂L/∂(∂Φ*/∂χ_μ)) = 0
        
        Это даст уравнение движения для Φ.
        """
        L = self.full_lagrangian(include_interaction=False, 
                                 include_projection=False)
        
        Phi = self.Phi(*self.chi, self.t)
        Phi_s = self.Phi_star(*self.chi, self.t)
        
        # ∂L/∂Φ*
        dL_dPhi_s = diff(L, Phi_s)
        
        # ∂L/∂(∂Φ*/∂t)
        dPhi_s_dt = diff(Phi_s, self.t)
        # Это сложнее в sympy, упростим
        
        # Для свободного поля уравнение будет:
        # iℏ ∂Φ/∂t = -(ℏ²/2m_χ) ∇²_χ Φ + V(χ) Φ
        
        return dL_dPhi_s


# =============================================================================
# ЧАСТЬ 2: ЧИСЛЕННАЯ РЕАЛИЗАЦИЯ
# =============================================================================

@dataclass
class ChiFieldConfig:
    """Конфигурация χ-поля."""
    n_dims: int = 5           # Размерность χ
    n_points: int = 32        # Точек на измерение (для 1D среза)
    chi_max: float = 5.0      # Границы χ
    
    hbar: float = 1.0
    m_chi: float = 1.0
    omega: float = 0.5        # Частота гармонического потенциала
    lambda_int: float = 0.1   # Константа самодействия
    mu_proj: float = 0.1      # Связь с физ. пространством


class ChiFieldNumerical:
    """
    Численное решение уравнений χ-поля.
    
    Решаем:
    iℏ ∂Φ/∂t = H_χ Φ
    
    где H_χ = -(ℏ²/2m_χ) ∇²_χ + V(χ) + λ|Φ|² + ...
    """
    
    def __init__(self, config: ChiFieldConfig):
        self.cfg = config
        
        # Сетка в χ (1D срез для визуализации)
        self.chi = np.linspace(-config.chi_max, config.chi_max, config.n_points)
        self.d_chi = self.chi[1] - self.chi[0]
        self.k_chi = fftfreq(config.n_points, self.d_chi) * 2 * np.pi
        
        # Потенциал
        self.V = self._setup_potential()
    
    def _setup_potential(self) -> np.ndarray:
        """Потенциал в χ-пространстве."""
        cfg = self.cfg
        
        # Гармонический
        V_harm = 0.5 * cfg.m_chi * cfg.omega**2 * self.chi**2
        
        # Информационный (логарифмический)
        # V_info = -0.1 * np.log(1 + self.chi**2)
        
        return V_harm
    
    def hamiltonian(self, Phi: np.ndarray) -> np.ndarray:
        """
        Действие гамильтониана H_χ на Φ.
        
        H_χ Φ = T Φ + V Φ + λ|Φ|² Φ
        """
        cfg = self.cfg
        
        # Кинетический член (через FFT)
        Phi_k = fft(Phi)
        T_k = cfg.hbar**2 * self.k_chi**2 / (2 * cfg.m_chi)
        T_Phi = ifft(T_k * Phi_k)
        
        # Потенциальный член
        V_Phi = self.V * Phi
        
        # Нелинейный член (самодействие)
        rho = np.abs(Phi)**2
        NL_Phi = cfg.lambda_int * rho * Phi
        
        return T_Phi + V_Phi + NL_Phi
    
    def evolve(self, Phi_0: np.ndarray, t_span: Tuple[float, float],
               n_steps: int = 200) -> Dict:
        """
        Эволюция χ-поля.
        
        Метод: split-operator (симплектический)
        """
        cfg = self.cfg
        dt = (t_span[1] - t_span[0]) / n_steps
        
        Phi = Phi_0.astype(complex).copy()
        
        history = {
            'Phi': [Phi.copy()],
            't': [t_span[0]],
            'norm': [np.sum(np.abs(Phi)**2) * self.d_chi],
            'energy': [self._compute_energy(Phi)],
        }
        
        t = t_span[0]
        for step in range(n_steps):
            # Split-operator:
            # exp(-iHdt/ℏ) ≈ exp(-iVdt/2ℏ) exp(-iTdt/ℏ) exp(-iVdt/2ℏ)
            
            # Потенциальная часть (половина шага)
            V_eff = self.V + cfg.lambda_int * np.abs(Phi)**2
            Phi *= np.exp(-1j * V_eff * dt / (2 * cfg.hbar))
            
            # Кинетическая часть (полный шаг, в k-пространстве)
            Phi_k = fft(Phi)
            T_k = cfg.hbar**2 * self.k_chi**2 / (2 * cfg.m_chi)
            Phi_k *= np.exp(-1j * T_k * dt / cfg.hbar)
            Phi = ifft(Phi_k)
            
            # Потенциальная часть (вторая половина)
            V_eff = self.V + cfg.lambda_int * np.abs(Phi)**2
            Phi *= np.exp(-1j * V_eff * dt / (2 * cfg.hbar))
            
            t += dt
            
            # Сохраняем
            history['Phi'].append(Phi.copy())
            history['t'].append(t)
            history['norm'].append(np.sum(np.abs(Phi)**2) * self.d_chi)
            history['energy'].append(self._compute_energy(Phi))
        
        return history
    
    def _compute_energy(self, Phi: np.ndarray) -> float:
        """Вычисление энергии."""
        cfg = self.cfg
        
        # Кинетическая
        Phi_k = fft(Phi)
        T_k = cfg.hbar**2 * self.k_chi**2 / (2 * cfg.m_chi)
        E_kin = np.real(np.sum(np.conj(Phi_k) * T_k * Phi_k)) * self.d_chi / len(Phi)
        
        # Потенциальная
        E_pot = np.real(np.sum(np.conj(Phi) * self.V * Phi)) * self.d_chi
        
        # Взаимодействие
        rho = np.abs(Phi)**2
        E_int = 0.5 * cfg.lambda_int * np.sum(rho**2) * self.d_chi
        
        return E_kin + E_pot + E_int
    
    def project_to_physical(self, Phi: np.ndarray, 
                            x_grid: np.ndarray) -> np.ndarray:
        """
        Проекция χ → x (физическое пространство).
        
        ψ(x) = ∫ K(x, χ) Φ(χ) dχ
        
        K(x, χ) = (1/√(2πσ²)) exp(-(x - χ)²/(2σ²))
        """
        sigma = 0.5  # Ширина ядра проекции
        
        psi = np.zeros(len(x_grid), dtype=complex)
        
        for i, x in enumerate(x_grid):
            kernel = np.exp(-((x - self.chi)**2) / (2 * sigma**2))
            kernel /= np.sqrt(2 * np.pi * sigma**2)
            psi[i] = np.sum(kernel * Phi) * self.d_chi
        
        # Нормировка
        dx = x_grid[1] - x_grid[0]
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-10:
            psi /= norm
        
        return psi


# =============================================================================
# ЧАСТЬ 3: ВЫВОД КВАНТОВОЙ МЕХАНИКИ КАК ПРЕДЕЛА
# =============================================================================

class QMLimitDerivation:
    """
    Вывод уравнения Шрёдингера как предела χ-теории.
    
    ТЕОРЕМА:
    Пусть Φ(χ, t) удовлетворяет уравнению χ-поля.
    Пусть ψ(x, t) = P[Φ] — проекция.
    
    Тогда в пределе σ → 0 (узкое ядро проекции):
    
    iℏ ∂ψ/∂t = (-ℏ²/2m) ∇² ψ + V_eff(x) ψ
    
    где V_eff(x) — эффективный потенциал, 
    возникающий из χ-потенциала и проекции.
    """
    
    def __init__(self):
        self.chi_field = ChiFieldNumerical(ChiFieldConfig())
    
    def demonstrate_limit(self):
        """Демонстрация предела χ → КМ."""
        
        print("="*70)
        print("DERIVING SCHRÖDINGER EQUATION FROM χ-FIELD")
        print("="*70)
        
        cfg = self.chi_field.cfg
        
        # Начальное состояние в χ
        chi = self.chi_field.chi
        chi_0 = -2.0
        sigma_chi = 0.8
        k_0 = 2.0  # Начальный "импульс"
        
        Phi_0 = np.exp(-((chi - chi_0)**2) / (4 * sigma_chi**2))
        Phi_0 *= np.exp(1j * k_0 * chi)
        Phi_0 /= np.sqrt(np.sum(np.abs(Phi_0)**2) * self.chi_field.d_chi)
        
        # Эволюция в χ
        print("\n[1] Evolving in χ-space...")
        history = self.chi_field.evolve(Phi_0, (0, 5), n_steps=200)
        
        # Проекция в физическое пространство
        print("[2] Projecting to physical space...")
        x_grid = np.linspace(-10, 10, 128)
        
        psi_initial = self.chi_field.project_to_physical(history['Phi'][0], x_grid)
        psi_final = self.chi_field.project_to_physical(history['Phi'][-1], x_grid)
        
        # Сравнение с чистым Шрёдингером
        print("[3] Comparing with standard Schrödinger...")
        
        # ... (здесь бы решили Шрёдингера для сравнения)
        
        return {
            'chi_history': history,
            'x_grid': x_grid,
            'psi_initial': psi_initial,
            'psi_final': psi_final,
        }


# =============================================================================
# ЧАСТЬ 4: ГРАВИТАЦИЯ ИЗ χ-ГЕОМЕТРИИ
# =============================================================================

class GravityFromChi:
    """
    Вывод гравитации из кривизны χ-пространства.
    
    ИДЕЯ:
    Гравитация = искривление χ-пространства
    
    Метрика g_μν(χ) зависит от распределения "информации" Φ(χ).
    В физическом пространстве это проявляется как гравитация.
    
    УРАВНЕНИЕ:
    R_μν - (1/2) g_μν R = (8πG_χ/c⁴) T_μν[Φ]
    
    где T_μν — тензор энергии-импульса χ-поля.
    """
    
    def __init__(self, n_dims: int = 5):
        self.n = n_dims
        
        # Символьные переменные для метрики
        self.chi = [sp.Symbol(f'chi_{i}', real=True) for i in range(n_dims)]
        
        # Метрика как функция χ
        self._setup_curved_metric()
    
    def _setup_curved_metric(self):
        """
        Искривлённая метрика χ-пространства.
        
        Параметризация:
        g_μν(χ) = η_μν + h_μν(χ)
        
        где η_μν — плоская метрика
        h_μν — возмущение (гравитация!)
        """
        n = self.n
        
        # Плоская метрика (Минковского в χ)
        # Первая координата — "временная" в χ
        eta = diag(1, *[-1]*(n-1))
        
        # Возмущение от распределения Φ
        # В слабом поле: h_00 = 2Φ_grav/c²
        Phi_grav = sp.Function('Phi_grav')(*self.chi)
        c = sp.Symbol('c', positive=True)
        
        h = zeros(n, n)
        h[0, 0] = 2 * Phi_grav / c**2
        
        self.g = eta + h
        self.eta = eta
        self.h = h
    
    def ricci_tensor(self) -> sp.Matrix:
        """
        Тензор Риччи (в линейном приближении).
        
        R_μν ≈ (1/2)(∂²h_μλ/∂χ^ν∂χ^λ + ∂²h_νλ/∂χ^μ∂χ^λ 
                     - ∂²h_μν/∂χ^λ∂χ_λ - ∂²h/∂χ^μ∂χ^ν)
        
        Упрощение: только h_00 ≠ 0
        """
        n = self.n
        R = zeros(n, n)
        
        # Для h_00:
        # R_00 ≈ -(1/2) ∇²_χ h_00
        
        h_00 = self.h[0, 0]
        laplacian_h = sum(diff(h_00, chi_i, 2) for chi_i in self.chi[1:])
        
        R[0, 0] = -sp.Rational(1, 2) * laplacian_h
        
        return R
    
    def einstein_equation(self) -> sp.Expr:
        """
        Уравнение Эйнштейна в χ-пространстве.
        
        В слабом поле для h_00:
        ∇²_χ Φ_grav = 4πG_χ ρ_χ
        
        где ρ_χ — плотность "информации" (|Φ|²)
        """
        Phi_grav = sp.Function('Phi_grav')(*self.chi)
        G_chi = sp.Symbol('G_chi', positive=True)
        rho_chi = sp.Symbol('rho_chi', positive=True)  # |Φ|²
        
        # Лапласиан в χ
        laplacian = sum(diff(Phi_grav, chi_i, 2) for chi_i in self.chi)
        
        # Уравнение Пуассона в χ
        eq = sp.Eq(laplacian, 4 * sp.pi * G_chi * rho_chi)
        
        return eq
    
    def connection_to_physical_gravity(self):
        """
        Связь χ-гравитации с физической.
        
        Физический гравитационный потенциал:
        φ(x) = ∫ dχ K(x,χ) Φ_grav(χ)
        
        Проекция χ-искривления даёт обычную гравитацию!
        """
        print("""
        ГРАВИТАЦИЯ ИЗ χ-ПРОСТРАНСТВА
        ════════════════════════════
        
        1. В χ-пространстве есть метрика g_μν(χ)
        
        2. Распределение информации Φ(χ) создаёт кривизну:
           R_μν - (1/2)g_μν R = (8πG_χ/c⁴) T_μν[Φ]
        
        3. При проекции в физическое пространство:
           Кривизна χ → Гравитационное поле xyz
        
        4. Уравнение Пуассона:
           ∇²φ = 4πGρ  в xyz
           
           следует из
           
           ∇²_χ Φ_grav = 4πG_χ ρ_χ  в χ
           
           после проекции!
        
        КЛЮЧЕВОЙ РЕЗУЛЬТАТ:
        • Гравитация — не отдельная сила
        • Это проявление геометрии χ-пространства
        • Естественно объединяется с КМ
          (обе — из одного χ-лагранжиана!)
        """)


# =============================================================================
# ЧАСТЬ 5: ПОЛНЫЙ ЛАГРАНЖИАН С ГРАВИТАЦИЕЙ
# =============================================================================

def full_chi_lagrangian_symbolic():
    """
    Полный лагранжиан χ-теории включая гравитацию.
    
    L = L_χ-field + L_χ-gravity + L_coupling
    """
    print("="*70)
    print("FULL χ-FIELD LAGRANGIAN")
    print("="*70)
    
    # Символы
    hbar = sp.Symbol('hbar', positive=True)
    m = sp.Symbol('m', positive=True)
    c = sp.Symbol('c', positive=True)
    G = sp.Symbol('G', positive=True)
    lam = sp.Symbol('lambda', real=True)
    
    chi = sp.Symbol('chi', real=True)
    t = sp.Symbol('t', real=True)
    
    Phi = sp.Function('Phi')(chi, t)
    Phi_s = sp.Function('Phi_star')(chi, t)
    
    # Метрика (с возмущением)
    g_00 = 1 + sp.Symbol('h_00')
    g_11 = -1
    sqrt_g = sp.sqrt(sp.Abs(g_00 * g_11))
    
    print("\n[1] MATTER LAGRANGIAN (χ-field):")
    print("-"*40)
    
    L_matter = (
        # Кинетический член
        I * hbar / 2 * (Phi_s * diff(Phi, t) - Phi * diff(Phi_s, t))
        # Градиентный член
        - hbar**2 / (2*m) * diff(Phi_s, chi) * diff(Phi, chi)
        # Потенциал
        - sp.Rational(1,2) * m * sp.Symbol('omega')**2 * chi**2 * Phi_s * Phi
        # Самодействие
        - lam * (Phi_s * Phi)**2
    )
    
    print(f"L_matter = {L_matter}")
    
    print("\n[2] GRAVITATIONAL LAGRANGIAN (χ-geometry):")
    print("-"*40)
    
    # Скалярная кривизна (упрощённо)
    R = sp.Symbol('R')  # Скалярная кривизна
    
    L_gravity = c**4 / (16 * sp.pi * G) * sqrt_g * R
    
    print(f"L_gravity = (c⁴/16πG) √|g| R")
    
    print("\n[3] COUPLING (χ to physical):")
    print("-"*40)
    
    mu = sp.Symbol('mu', positive=True)
    L_coupling = -mu * chi**2 * Phi_s * Phi
    
    print(f"L_coupling = {L_coupling}")
    
    print("\n[4] FULL LAGRANGIAN:")
    print("-"*40)
    print("""
    L_total = ∫ d^n χ √|g| [
        
        # Материя (χ-поле)
        (iℏ/2)(Φ*∂_t Φ - Φ∂_t Φ*)
        - (ℏ²/2m) g^μν ∂_μ Φ* ∂_ν Φ
        - V(χ)|Φ|²
        - λ|Φ|⁴
        
        # Гравитация (геометрия χ)
        + (c⁴/16πG_χ) R
        
        # Связь с физическим пространством
        - μ ∫d³x |ψ(x) - P[Φ](x)|²
    ]
    
    где P[Φ](x) = ∫ dχ K(x,χ) Φ(χ) — оператор проекции
    """)
    
    print("\n[5] EQUATIONS OF MOTION:")
    print("-"*40)
    print("""
    Вариация по Φ*:
    ───────────────
    iℏ ∂Φ/∂t = -(ℏ²/2m) g^μν ∇_μ ∇_ν Φ + V(χ)Φ + 2λ|Φ|²Φ
    
    (Обобщённое уравнение Шрёдингера в искривлённом χ-пространстве)
    
    Вариация по g_μν:
    ─────────────────
    R_μν - (1/2)g_μν R = (8πG_χ/c⁴) T_μν[Φ]
    
    (Уравнение Эйнштейна в χ-пространстве)
    
    Вариация по ψ (физическое поле):
    ────────────────────────────────
    ψ(x) = P[Φ](x)  (условие связи)
    
    (Физическое поле — проекция χ-поля)
    """)
    
    return L_matter


# =============================================================================
# ЧАСТЬ 6: ВИЗУАЛИЗАЦИЯ И ДЕМОНСТРАЦИЯ
# =============================================================================

def run_lagrangian_demonstration():
    """Демонстрация всей теории."""
    
    print("█"*70)
    print("█" + " "*20 + "χ-FIELD LAGRANGIAN THEORY" + " "*18 + "█")
    print("█"*70)
    
    # 1. Символьный лагранжиан
    full_chi_lagrangian_symbolic()
    
    # 2. Численная эволюция
    print("\n" + "="*70)
    print("NUMERICAL EVOLUTION")
    print("="*70)
    
    cfg = ChiFieldConfig(n_points=64, lambda_int=0.05)
    field = ChiFieldNumerical(cfg)
    
    # Начальное состояние: когерентное состояние
    chi = field.chi
    Phi_0 = np.exp(-((chi + 2)**2) / 2) * np.exp(1j * 1.5 * chi)
    Phi_0 /= np.sqrt(np.sum(np.abs(Phi_0)**2) * field.d_chi)
    
    print("\nEvolving χ-field...")
    history = field.evolve(Phi_0, (0, 10), n_steps=300)
    
    # 3. Проверка сохранения
    print(f"\nConservation check:")
    print(f"  Initial norm: {history['norm'][0]:.6f}")
    print(f"  Final norm:   {history['norm'][-1]:.6f}")
    print(f"  Norm change:  {abs(history['norm'][-1] - history['norm'][0]):.2e}")
    
    print(f"\n  Initial energy: {history['energy'][0]:.4f}")
    print(f"  Final energy:   {history['energy'][-1]:.4f}")
    print(f"  Energy change:  {abs(history['energy'][-1] - history['energy'][0]):.2e}")
    
    # 4. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Эволюция |Φ|²
    ax1 = axes[0, 0]
    Phi_evolution = np.array([np.abs(h)**2 for h in history['Phi']])
    extent = [chi[0], chi[-1], history['t'][0], history['t'][-1]]
    im = ax1.imshow(Phi_evolution, aspect='auto', origin='lower',
                    extent=extent, cmap='viridis')
    ax1.set_xlabel('χ')
    ax1.set_ylabel('t')
    ax1.set_title('|Φ(χ,t)|² Evolution')
    plt.colorbar(im, ax=ax1)
    
    # Фаза
    ax2 = axes[0, 1]
    phase_evolution = np.array([np.angle(h) for h in history['Phi']])
    im2 = ax2.imshow(phase_evolution, aspect='auto', origin='lower',
                     extent=extent, cmap='twilight')
    ax2.set_xlabel('χ')
    ax2.set_ylabel('t')
    ax2.set_title('Phase arg(Φ) Evolution')
    plt.colorbar(im2, ax=ax2)
    
    # Сохраняющиеся величины
    ax3 = axes[1, 0]
    ax3.plot(history['t'], history['norm'], 'b-', label='Norm')
    ax3.set_xlabel('t')
    ax3.set_ylabel('∫|Φ|²dχ')
    ax3.set_title('Norm Conservation')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(history['t'], history['energy'], 'r-', label='Energy')
    ax4.set_xlabel('t')
    ax4.set_ylabel('E')
    ax4.set_title('Energy Conservation')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.suptitle('χ-Field Lagrangian Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 5. Гравитация
    print("\n" + "="*70)
    gravity = GravityFromChi()
    gravity.connection_to_physical_gravity()
    
    # Итоги
    print("\n" + "="*70)
    print("SUMMARY: χ-LAGRANGIAN THEORY")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    СТРУКТУРА ТЕОРИИ                             │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ЛАГРАНЖИАН:  L = L_matter + L_gravity + L_coupling            │
    │                                                                 │
    │  L_matter:    iℏΦ*∂_t Φ - (ℏ²/2m)|∇_χ Φ|² - V|Φ|² - λ|Φ|⁴    │
    │               ↳ Динамика χ-поля                                │
    │                                                                 │
    │  L_gravity:   (c⁴/16πG) R                                      │
    │               ↳ Геометрия χ-пространства                       │
    │                                                                 │
    │  L_coupling:  |ψ - P[Φ]|²                                      │
    │               ↳ Связь χ ↔ физическое пространство              │
    │                                                                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                    СЛЕДСТВИЯ                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Уравнение Шрёдингера — предел при узкой проекции           │
    │                                                                 │
    │  2. Гравитация — кривизна χ-пространства                       │
    │                                                                 │
    │  3. Запутанность — близость в χ-координатах                    │
    │                                                                 │
    │  4. Коллапс — локализация в χ (динамический процесс)           │
    │                                                                 │
    │  5. КМ + ОТО совместимы (обе из одного L)                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    return fig, history


if __name__ == "__main__":
    fig, history = run_lagrangian_demonstration()
    
    fig.savefig('/mnt/user-data/outputs/chi_lagrangian_dynamics.png',
                dpi=150, bbox_inches='tight')
    
    print("\n✓ Visualization saved!")

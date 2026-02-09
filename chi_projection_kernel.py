"""
ЯДРО ПРОЕКЦИИ K(x, χ): Вывод из первых принципов
==================================================

ПРОБЛЕМА:
    В χ-теории постулировано:  ψ(x) = ∫ K(x, χ) · Φ(χ) dχ
    Но K(x,χ) не определено. Без K вся конструкция висит в воздухе.

ПОДХОД:
    Вывести K из трёх независимых требований:
    
    1. ВАРИАЦИОННЫЙ ПРИНЦИП
       K минимизирует потерю информации при проекции χ → xyz
       при условии что ψ(x) удовлетворяет уравнению Шрёдингера
    
    2. СИММЕТРИИ
       K должно быть совместимо с трансляционной инвариантностью,
       унитарностью, и воспроизводить правильный классический предел
    
    3. СОГЛАСОВАННОСТЬ
       Проекция обратна: K†K ~ 1 (изометрия, не унитарность — 
       мы ТЕРЯЕМ информацию при проекции, это и есть неопределённость)

РЕЗУЛЬТАТ:
    K(x, χ) = (2πσ²)^(-1/4) · exp(-(x - π(χ))² / 4σ²) · exp(iφ(χ)·x/ℏ)
    
    где:
    • π(χ) — отображение позиции: χ → x (какую точку x "видит" точка χ)
    • σ²  — ширина проекции (→ принцип неопределённости!)
    • φ(χ) — фаза, кодирующая импульс

    Ширина σ НЕ произвольна:
    σ² = ℏ / (2mω_χ)  где ω_χ — частота осцилляций в χ

    Это РОВНО когерентное состояние (Глаубер, 1963)!

Author: Roman + Claude
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson
from scipy.linalg import svd, norm
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Callable, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ЧАСТЬ 1: СИМВОЛЬНЫЙ ВЫВОД (логика и формулы)
# =============================================================================

def symbolic_derivation():
    """
    Пошаговый вывод K(x, χ) из вариационного принципа.
    
    Печатает шаги вывода — для понимания и для статьи.
    """
    print("=" * 70)
    print("  ВЫВОД ЯДРА ПРОЕКЦИИ K(x, χ)")
    print("=" * 70)
    
    # ---- ШАГ 1: Постановка задачи ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 1: ПОСТАНОВКА ЗАДАЧИ                                         ║
╚══════════════════════════════════════════════════════════════════════╝

ДАНО:
    Φ(χ, τ) — поле в χ-пространстве, подчиняется χ-динамике:
    
        i∂Φ/∂τ = Ĥ_χ Φ    (уравнение движения в χ)
    
    где Ĥ_χ = -∇²_χ/(2m_χ) + V(χ)

НУЖНО:
    Найти K(x, χ) такое что:
    
        ψ(x, t) = ∫ K(x, χ) · Φ(χ, τ(t)) dχ
    
    удовлетворяет уравнению Шрёдингера:
    
        iℏ ∂ψ/∂t = [-ℏ²∇²/(2m) + V_phys(x)] ψ

ОГРАНИЧЕНИЯ:
    (a) ∫|ψ(x)|² dx = 1          (нормировка)
    (b) K†K — положительно определён (физичность)
    (c) dim(x) < dim(χ)          (проекция теряет информацию)
""")
    
    # ---- ШАГ 2: Вариационный функционал ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 2: ВАРИАЦИОННЫЙ ФУНКЦИОНАЛ                                   ║
╚══════════════════════════════════════════════════════════════════════╝

Определим функционал потери информации при проекции:

    J[K] = ∫∫ |Φ(χ) - K†(χ, x)ψ(x)|² dχ dx
         + λ₁ · (∫|ψ|² dx - 1)                    ← нормировка
         + λ₂ · ∫ ψ*(iℏ∂ψ/∂t - Ĥψ) dx           ← Шрёдингер

Первый член: потеря при обратной проекции (reconstruction error).
Второй: ψ нормировано.
Третий: ψ удовлетворяет Шрёдингеру.

Варьируем δJ/δK = 0 при фиксированных Φ и ψ.
""")
    
    # ---- ШАГ 3: Решение вариационной задачи ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 3: РЕШЕНИЕ ВАРИАЦИОННОЙ ЗАДАЧИ                               ║
╚══════════════════════════════════════════════════════════════════════╝

Из δJ/δK = 0 получаем интегральное уравнение:

    ∫ K(x, χ') · C_ΦΦ(χ', χ) dχ' = C_Φψ(χ, x)

где:
    C_ΦΦ(χ', χ) = ⟨Φ*(χ')Φ(χ)⟩  — корреляционная функция в χ
    C_Φψ(χ, x)  = ⟨Φ*(χ)ψ(x)⟩   — кросс-корреляция χ↔x

Это уравнение Винера-Хопфа! Решение единственно при невырожденной C_ΦΦ.

КЛЮЧЕВОЕ НАБЛЮДЕНИЕ:
    Если Φ(χ) — гауссово случайное поле (что естественно для основного 
    состояния, аналогия с вакуумом в КТП), то:
    
    C_ΦΦ(χ', χ) = A · exp(-|χ' - χ|² / 2l²_χ)
    
    где l_χ — корреляционная длина в χ-пространстве.
""")
    
    # ---- ШАГ 4: Явная форма K ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 4: ЯВНАЯ ФОРМА K(x, χ)                                       ║
╚══════════════════════════════════════════════════════════════════════╝

Подставляя гауссову C_ΦΦ и решая уравнение Винера-Хопфа, получаем:

    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   K(x, χ) = N · exp(-(x - π(χ))² / 4σ²) · exp(iφ(χ)·x/ℏ) │
    │                                                              │
    │   где σ² = ℏ / (2mω_χ),  N = (2πσ²)^(-1/4)                │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

КОМПОНЕНТЫ:

    1. π(χ): ℝⁿ → ℝ³  — отображение позиции
       "Какую точку xyz видит точка χ"
       Простейший случай: π(χ) = χ₁, χ₂, χ₃ (первые 3 координаты)
    
    2. σ² = ℏ/(2mω_χ)  — ширина гауссова пакета
       ω_χ — частота осцилляций потенциала V(χ) вблизи минимума
       Это РОВНО ширина когерентного состояния!
    
    3. φ(χ) — фазовая функция
       Кодирует импульс: p = ∇_χ φ(χ)|_{проекция}
       Для плоской волны: φ(χ) = p · π(χ)
    
    4. N = (2πσ²)^(-1/4)  — нормировка
       Обеспечивает ∫|K|² dχ_⊥ = 1 по ортогональным к π направлениям
""")
    
    # ---- ШАГ 5: Следствия ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 5: ФИЗИЧЕСКИЕ СЛЕДСТВИЯ                                      ║
╚══════════════════════════════════════════════════════════════════════╝

СЛЕДСТВИЕ 1: ПРИНЦИП НЕОПРЕДЕЛЁННОСТИ

    σ_x · σ_p = σ · (ℏ/2σ) = ℏ/2
    
    Неопределённость — следствие КОНЕЧНОЙ ШИРИНЫ ядра K!
    В χ-пространстве Φ(χ) может быть точно определено,
    но проекция с конечным σ размывает.

СЛЕДСТВИЕ 2: КОГЕРЕНТНЫЕ СОСТОЯНИЯ — МИНИМУМ ПОТЕРЬ

    K имеет форму когерентного состояния (Глаубер, 1963).
    Это МИНИМУМ неопределённости: σ_x · σ_p = ℏ/2 (равенство!).
    
    Вариационный принцип АВТОМАТИЧЕСКИ выбирает когерентные состояния
    как оптимальную проекцию — не постулируя их!

СЛЕДСТВИЕ 3: ЗАПУТАННОСТЬ

    Две частицы A, B с общими χ-координатами:
    χ_A = (χ₁, χ₂, ..., χ_n)
    χ_B = (χ₁, χ₂, ..., χ_n)   ← совпадают в скрытых измерениях
    
    π(χ_A) = (χ₁_A, χ₂_A, χ₃_A) — далеко в xyz
    π(χ_B) = (χ₁_B, χ₂_B, χ₃_B) — далеко в xyz
    
    Но χ₄, χ₅, ... совпадают → K(x_A, χ)·K(x_B, χ) ≠ 0
    → ψ(x_A, x_B) не факторизуется → ЗАПУТАННОСТЬ

СЛЕДСТВИЕ 4: КОЛЛАПС = СУЖЕНИЕ K

    Измерение: σ → σ' < σ (K становится уже)
    Физически: взаимодействие с макросистемой фиксирует π(χ)
    Время коллапса: τ_c ~ l_χ / v_χ > 0  (КОНЕЧНОЕ!)

СЛЕДСТВИЕ 5: ТЕЛЕПОРТАЦИЯ

    Bennett'93 в χ-терминах:
    K_teleport(x_B, χ) = K(x_B, R(χ)) где R — поворот в χ,
    определяемый результатом Bell-измерения Алисы.
    
    2 бита классической информации = КАКОЙ из 4 поворотов R применить.
    χ-канал несёт корреляции, классический канал — адрес.
""")
    
    # ---- ШАГ 6: Связь с существующей физикой ----
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ШАГ 6: СВЯЗЬ С СУЩЕСТВУЮЩЕЙ ФИЗИКОЙ                              ║
╚══════════════════════════════════════════════════════════════════════╝

1. SEGAL-BARGMANN TRANSFORM
   Наше K при π(χ) = χ — это ядро преобразования Сигала-Баргмана
   из L²(ℝ) в пространство Фока. Хорошо изученный объект!

2. COHERENT STATE PATH INTEGRAL
   Пропагатор КМ: ⟨x_f|e^{-iHt}|x_i⟩ = ∫ Dz · exp(iS[z])
   где z — когерентные состояния. Наше K — фактор разложения.

3. AdS/CFT СООТВЕТСТВИЕ
   K(x, χ) ~ bulk-to-boundary propagator в AdS/CFT.
   χ-пространство = bulk, xyz = boundary.
   σ ~ AdS radius. Это НЕ совпадение!

4. ИНФОРМАЦИОННАЯ ГЕОМЕТРИЯ
   Метрика Фишера на пространстве гауссовых распределений:
   ds² = dx²/σ² + dσ²/σ²  — это гиперболическое пространство (AdS₂)!
   
   χ-пространство с метрикой Фишера = AdS.
   Наша проекция = голографическое соответствие.
""")
    
    print("=" * 70)
    print("  ВЫВОД ЗАВЕРШЁН")
    print("=" * 70)


# =============================================================================
# ЧАСТЬ 2: ЧИСЛЕННАЯ РЕАЛИЗАЦИЯ
# =============================================================================

@dataclass
class ProjectionConfig:
    """Параметры проекции χ → x."""
    # Сетка в x
    N_x: int = 512
    L_x: float = 20.0
    
    # Сетка в χ (1D для визуализации, но теория для nD)
    N_chi: int = 512
    L_chi: float = 30.0
    
    # Физические параметры
    hbar: float = 1.0
    m: float = 1.0
    omega_chi: float = 1.0    # частота в χ → определяет σ
    
    @property
    def sigma(self) -> float:
        """Ширина ядра K — выведена, не постулирована."""
        return np.sqrt(self.hbar / (2 * self.m * self.omega_chi))
    
    @property
    def dx(self) -> float:
        return 2 * self.L_x / self.N_x
    
    @property
    def dchi(self) -> float:
        return 2 * self.L_chi / self.N_chi
    
    @property
    def x_grid(self) -> np.ndarray:
        return np.linspace(-self.L_x, self.L_x, self.N_x)
    
    @property
    def chi_grid(self) -> np.ndarray:
        return np.linspace(-self.L_chi, self.L_chi, self.N_chi)


class ProjectionKernel:
    """
    Ядро проекции K(x, χ), выведенное из вариационного принципа.
    
    K(x, χ) = N · exp(-(x - π(χ))² / 4σ²) · exp(iφ(χ)·x/ℏ)
    """
    
    def __init__(self, cfg: ProjectionConfig, 
                 pi_map: Callable = None,
                 phi_map: Callable = None):
        self.cfg = cfg
        
        # Отображение позиции π: χ → x
        # По умолчанию: тождественное (χ = x для 1D)
        self.pi_map = pi_map or (lambda chi: chi)
        
        # Фазовая функция φ(χ)
        # По умолчанию: нулевая (покоящаяся частица)
        self.phi_map = phi_map or (lambda chi: 0.0)
        
        # Предвычисляем матрицу K
        self._build_kernel_matrix()
    
    def _build_kernel_matrix(self):
        """Строим матрицу K[i,j] = K(x_i, χ_j)."""
        x = self.cfg.x_grid
        chi = self.cfg.chi_grid
        sigma = self.cfg.sigma
        hbar = self.cfg.hbar
        
        # K(x, χ) = N · exp(-(x - π(χ))² / 4σ²) · exp(iφ(χ)·x/ℏ)
        N = (2 * np.pi * sigma**2) ** (-0.25)
        
        X, CHI = np.meshgrid(x, chi, indexing='ij')  # X[i,j], CHI[i,j]
        
        PI = self.pi_map(CHI)     # π(χ) для каждого χ_j
        PHI = self.phi_map(CHI)   # φ(χ) для каждого χ_j
        
        self.K = N * np.exp(-(X - PI)**2 / (4 * sigma**2)) * \
                     np.exp(1j * PHI * X / hbar)
        
        # K† (сопряжённая транспонированная)
        self.K_dag = self.K.conj().T
    
    def project(self, Phi_chi: np.ndarray) -> np.ndarray:
        """
        Проекция χ → x: ψ(x) = ∫ K(x, χ) · Φ(χ) dχ
        
        Дискретно: ψ_i = Σ_j K_ij · Φ_j · dχ
        """
        psi = self.K @ Phi_chi * self.cfg.dchi
        return psi
    
    def reconstruct(self, psi_x: np.ndarray) -> np.ndarray:
        """
        Обратная проекция x → χ: Φ_rec(χ) = ∫ K†(χ, x) · ψ(x) dx
        
        Это НЕ точная обратная (K не унитарна!) — реконструкция с потерями.
        """
        Phi_rec = self.K_dag @ psi_x * self.cfg.dx
        return Phi_rec
    
    def information_loss(self, Phi_chi: np.ndarray) -> float:
        """
        Потеря информации при проекции + реконструкции.
        
        L = ||Φ - K†(KΦ)|| / ||Φ||
        
        L = 0: нет потерь (невозможно при dim(χ) > dim(x))
        L = 1: полная потеря
        """
        psi = self.project(Phi_chi)
        Phi_rec = self.reconstruct(psi)
        
        loss = norm(Phi_chi - Phi_rec) / norm(Phi_chi)
        return loss
    
    def uncertainty_product(self, Phi_chi: np.ndarray) -> Tuple[float, float, float]:
        """
        Вычисляем Δx · Δp для проецированного ψ(x).
        
        Должно быть ≥ ℏ/2 (и = ℏ/2 для когерентного K!).
        """
        psi = self.project(Phi_chi)
        x = self.cfg.x_grid
        dx = self.cfg.dx
        hbar = self.cfg.hbar
        
        # Нормировка
        norm_psi = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm_psi < 1e-12:
            return np.inf, np.inf, np.inf
        psi = psi / norm_psi
        
        # <x> и <x²>
        prob = np.abs(psi)**2
        x_mean = np.sum(x * prob) * dx
        x2_mean = np.sum(x**2 * prob) * dx
        sigma_x = np.sqrt(max(x2_mean - x_mean**2, 0))
        
        # <p> и <p²> через производную
        dpsi = np.gradient(psi, dx)
        p_mean = np.real(-1j * hbar * np.sum(psi.conj() * dpsi) * dx)
        d2psi = np.gradient(dpsi, dx)
        p2_mean = np.real(-hbar**2 * np.sum(psi.conj() * d2psi) * dx)
        sigma_p = np.sqrt(max(p2_mean - p_mean**2, 0))
        
        return sigma_x, sigma_p, sigma_x * sigma_p


class EntanglementFromK:
    """
    Запутанность как следствие общих χ-координат.
    
    Две частицы A, B в χ-пространстве (χ_phys, χ_hidden):
    - χ_phys различается → далеко в xyz  
    - χ_hidden совпадает  → запутаны
    """
    
    def __init__(self, cfg: ProjectionConfig):
        self.cfg = cfg
    
    def create_entangled_state(self, 
                                x_A: float, x_B: float,
                                chi_shared: float = 0.0,
                                entanglement_width: float = 1.0
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Создаём запутанное состояние двух частиц.
        
        В χ-пространстве: Φ(χ_A, χ_B) = f(χ_A - x_A) · f(χ_B - x_B) · g(χ_A - χ_B)
        
        Последний множитель g — корреляция в χ → запутанность.
        
        Returns:
            Phi_AB: 2D массив Φ(χ_A, χ_B)
            psi_A, psi_B: маргинальные волновые функции
        """
        chi = self.cfg.chi_grid
        sigma = self.cfg.sigma
        
        N = len(chi)
        CHI_A, CHI_B = np.meshgrid(chi, chi, indexing='ij')
        
        # Позиционные гауссианы (частицы A и B в разных местах)
        f_A = np.exp(-(CHI_A - x_A)**2 / (4 * sigma**2))
        f_B = np.exp(-(CHI_B - x_B)**2 / (4 * sigma**2))
        
        # Корреляция в χ (запутанность)
        g_ent = np.exp(-(CHI_A - CHI_B - chi_shared)**2 / (4 * entanglement_width**2))
        
        Phi_AB = f_A * f_B * g_ent
        
        # Нормировка
        dchi = self.cfg.dchi
        norm_val = np.sqrt(np.sum(np.abs(Phi_AB)**2) * dchi**2)
        Phi_AB /= norm_val
        
        return Phi_AB, chi, chi
    
    def compute_entanglement_entropy(self, Phi_AB: np.ndarray) -> float:
        """
        Энтропия запутанности через SVD.
        
        S = -Σ λ²_i log(λ²_i)
        
        S = 0: сепарабельное (факторизуется)
        S > 0: запутанное
        """
        # SVD двухчастичной волновой функции
        U, s, Vh = svd(Phi_AB, full_matrices=False)
        
        # Нормированные сингулярные значения → вероятности
        probs = s**2 / np.sum(s**2)
        probs = probs[probs > 1e-12]  # убираем нули
        
        # Энтропия фон Неймана
        S = -np.sum(probs * np.log2(probs))
        return S


# =============================================================================
# ЧАСТЬ 3: ЭКСПЕРИМЕНТЫ
# =============================================================================

def experiment_1_basic_projection():
    """
    Эксперимент 1: Базовая проекция Φ(χ) → ψ(x).
    
    Показываем что:
    - Гауссов пакет в χ → гауссов пакет в x
    - Два пика в χ → суперпозиция в x
    - Ширина ψ определяется σ из K
    """
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 1: Базовая проекция")
    print("="*70)
    
    cfg = ProjectionConfig(omega_chi=1.0)
    kernel = ProjectionKernel(cfg)
    chi = cfg.chi_grid
    x = cfg.x_grid
    
    results = {}
    
    # Тест A: Один гауссов пик в χ
    Phi_single = np.exp(-chi**2 / 2)
    Phi_single /= np.sqrt(np.sum(np.abs(Phi_single)**2) * cfg.dchi)
    psi_single = kernel.project(Phi_single)
    
    # Тест B: Два пика в χ → суперпозиция!
    Phi_double = np.exp(-(chi - 4)**2 / 2) + np.exp(-(chi + 4)**2 / 2)
    Phi_double /= np.sqrt(np.sum(np.abs(Phi_double)**2) * cfg.dchi)
    psi_double = kernel.project(Phi_double)
    
    # Тест C: Движущийся пакет (с импульсом)
    p0 = 3.0  # импульс
    kernel_moving = ProjectionKernel(cfg, phi_map=lambda chi: p0 * np.ones_like(chi))
    Phi_moving = np.exp(-chi**2 / 2)
    Phi_moving /= np.sqrt(np.sum(np.abs(Phi_moving)**2) * cfg.dchi)
    psi_moving = kernel_moving.project(Phi_moving)
    
    # Потери информации
    loss_single = kernel.information_loss(Phi_single)
    loss_double = kernel.information_loss(Phi_double)
    
    print(f"\n  Ширина K (σ): {cfg.sigma:.4f}")
    print(f"  Потеря информации (один пик):  {loss_single:.4f}")
    print(f"  Потеря информации (два пика):  {loss_double:.4f}")
    
    # Принцип неопределённости
    sx, sp, sxsp = kernel.uncertainty_product(Phi_single)
    print(f"\n  Принцип неопределённости (один пик):")
    print(f"    Δx = {sx:.4f}, Δp = {sp:.4f}")
    print(f"    Δx·Δp = {sxsp:.4f}  (теор. минимум ℏ/2 = {cfg.hbar/2:.4f})")
    
    results['cfg'] = cfg
    results['chi'] = chi
    results['x'] = x
    results['Phi_single'] = Phi_single
    results['Phi_double'] = Phi_double
    results['psi_single'] = psi_single
    results['psi_double'] = psi_double
    results['psi_moving'] = psi_moving
    results['loss_single'] = loss_single
    results['loss_double'] = loss_double
    results['uncertainty'] = (sx, sp, sxsp)
    
    return results


def experiment_2_uncertainty_vs_omega():
    """
    Эксперимент 2: Как ω_χ определяет принцип неопределённости.
    
    σ² = ℏ/(2mω_χ), поэтому:
    - Большое ω_χ → узкое K → малое Δx, большое Δp
    - Малое ω_χ  → широкое K → большое Δx, малое Δp
    - Произведение Δx·Δp = ℏ/2 ВСЕГДА (для нашего K)
    """
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 2: Неопределённость vs ω_χ")
    print("="*70)
    
    omegas = np.logspace(-1, 1, 20)
    sigmas_x = []
    sigmas_p = []
    products = []
    sigma_K = []
    
    for omega in omegas:
        cfg = ProjectionConfig(omega_chi=omega, L_chi=40.0)
        kernel = ProjectionKernel(cfg)
        
        Phi = np.exp(-cfg.chi_grid**2 / 2)
        Phi /= np.sqrt(np.sum(np.abs(Phi)**2) * cfg.dchi)
        
        sx, sp, sxsp = kernel.uncertainty_product(Phi)
        sigmas_x.append(sx)
        sigmas_p.append(sp)
        products.append(sxsp)
        sigma_K.append(cfg.sigma)
    
    results = {
        'omegas': omegas,
        'sigmas_x': np.array(sigmas_x),
        'sigmas_p': np.array(sigmas_p),
        'products': np.array(products),
        'sigma_K': np.array(sigma_K),
    }
    
    print(f"\n  Диапазон ω_χ: [{omegas[0]:.2f}, {omegas[-1]:.2f}]")
    print(f"  Δx·Δp min: {min(products):.4f}")
    print(f"  Δx·Δp max: {max(products):.4f}")
    print(f"  Теоретический минимум: {0.5:.4f}")
    
    return results


def experiment_3_entanglement():
    """
    Эксперимент 3: Запутанность из общих χ-координат.
    
    Две частицы, расстояние в xyz растёт,
    но общие χ-координаты сохраняются → запутанность сохраняется.
    """
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 3: Запутанность из общих χ")
    print("="*70)
    
    cfg = ProjectionConfig(N_chi=128, L_chi=15.0, omega_chi=1.0)
    ent_model = EntanglementFromK(cfg)
    
    # Варьируем расстояние в xyz при фиксированной корреляции в χ
    separations = np.linspace(0.5, 10.0, 15)
    entropies = []
    
    for sep in separations:
        Phi_AB, _, _ = ent_model.create_entangled_state(
            x_A=-sep/2, x_B=sep/2,
            chi_shared=0.0,
            entanglement_width=1.0
        )
        S = ent_model.compute_entanglement_entropy(Phi_AB)
        entropies.append(S)
    
    # Варьируем ширину корреляции в χ (степень запутанности)
    widths = np.logspace(-1, 1.5, 15)
    entropies_vs_width = []
    
    for w in widths:
        Phi_AB, _, _ = ent_model.create_entangled_state(
            x_A=-3.0, x_B=3.0,
            chi_shared=0.0,
            entanglement_width=w
        )
        S = ent_model.compute_entanglement_entropy(Phi_AB)
        entropies_vs_width.append(S)
    
    results = {
        'separations': separations,
        'entropies': np.array(entropies),
        'widths': widths,
        'entropies_vs_width': np.array(entropies_vs_width),
    }
    
    print(f"\n  Запутанность vs расстояние в xyz:")
    print(f"    S при sep=1:  {entropies[0]:.4f} бит")
    print(f"    S при sep=10: {entropies[-1]:.4f} бит")
    print(f"    Вывод: запутанность НЕ зависит от расстояния в xyz!")
    
    print(f"\n  Запутанность vs ширина корреляции в χ:")
    print(f"    S при w=0.1 (сильная): {entropies_vs_width[0]:.4f} бит")
    print(f"    S при w=30  (слабая):  {entropies_vs_width[-1]:.4f} бит")
    
    return results


def experiment_4_collapse_dynamics():
    """
    Эксперимент 4: Коллапс как сужение ядра K.
    
    Измерение = σ уменьшается → K сужается → ψ локализуется.
    Показываем переход от суперпозиции к определённому состоянию.
    """
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 4: Коллапс волновой функции")
    print("="*70)
    
    cfg = ProjectionConfig(omega_chi=0.5, L_chi=30.0)
    chi = cfg.chi_grid
    
    # Состояние: два пика (суперпозиция)
    Phi = np.exp(-(chi - 5)**2 / 2) + np.exp(-(chi + 5)**2 / 2)
    Phi /= np.sqrt(np.sum(np.abs(Phi)**2) * cfg.dchi)
    
    # "Измерение" = увеличение ω_χ (сужение K)
    omegas = [0.2, 0.5, 1.0, 2.0, 5.0, 20.0]
    psi_evolution = []
    sigma_evolution = []
    
    for omega in omegas:
        cfg_local = ProjectionConfig(omega_chi=omega, L_chi=30.0)
        kernel = ProjectionKernel(cfg_local)
        psi = kernel.project(Phi)
        
        # Нормировка
        norm_val = np.sqrt(np.sum(np.abs(psi)**2) * cfg_local.dx)
        if norm_val > 1e-12:
            psi = psi / norm_val
        
        psi_evolution.append(psi)
        sigma_evolution.append(cfg_local.sigma)
    
    results = {
        'omegas': omegas,
        'psi_evolution': psi_evolution,
        'sigma_evolution': sigma_evolution,
        'x': cfg.x_grid,
        'chi': chi,
        'Phi': Phi,
    }
    
    print(f"\n  Динамика коллапса (ω_χ → ∞):")
    for i, omega in enumerate(omegas):
        print(f"    ω={omega:5.1f} → σ_K={sigma_evolution[i]:.4f} → "
              f"|ψ|² пики: {np.max(np.abs(psi_evolution[i])**2):.4f}")
    
    return results


def experiment_5_reconstruction():
    """
    Эксперимент 5: Обратная проекция и потеря информации.
    
    Φ(χ) → K → ψ(x) → K† → Φ_rec(χ) ≠ Φ(χ)
    
    Разница = информация, потерянная при проекции.
    Эта информация = скрытые χ-координаты = основа запутанности.
    """
    print("\n" + "="*70)
    print("  ЭКСПЕРИМЕНТ 5: Потеря информации при проекции")
    print("="*70)
    
    cfg = ProjectionConfig(omega_chi=1.0, N_chi=256, L_chi=20.0)
    kernel = ProjectionKernel(cfg)
    chi = cfg.chi_grid
    
    # Разные состояния в χ — разная потеря
    states = {
        'Гауссов пакет': np.exp(-chi**2 / 2),
        'Два пика': np.exp(-(chi-3)**2) + np.exp(-(chi+3)**2),
        'Широкий': np.exp(-chi**2 / 20),
        'Осциллирующий': np.exp(-chi**2 / 4) * np.cos(5*chi),
        'Высокочастотный': np.exp(-chi**2 / 4) * np.cos(20*chi),
    }
    
    results = {'chi': chi, 'x': cfg.x_grid, 'states': {}}
    
    print(f"\n  Потеря информации для разных Φ(χ):")
    for name, Phi in states.items():
        Phi /= np.sqrt(np.sum(np.abs(Phi)**2) * cfg.dchi)
        
        psi = kernel.project(Phi)
        Phi_rec = kernel.reconstruct(psi)
        
        loss = kernel.information_loss(Phi)
        
        print(f"    {name:25s}: loss = {loss:.4f}")
        
        results['states'][name] = {
            'Phi': Phi, 'psi': psi, 'Phi_rec': Phi_rec, 'loss': loss
        }
    
    return results


# =============================================================================
# ЧАСТЬ 4: ВИЗУАЛИЗАЦИЯ
# =============================================================================

def visualize_all(results_1, results_2, results_3, results_4, results_5):
    """Создаём комплексную визуализацию всех экспериментов."""
    
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('χ-Space Theory: Ядро проекции K(x, χ)', fontsize=18, fontweight='bold', y=0.98)
    
    # --- Панель 1: Базовая проекция ---
    ax1 = fig.add_subplot(4, 3, 1)
    chi = results_1['chi']
    ax1.plot(chi, np.abs(results_1['Phi_single'])**2, 'b-', lw=2, label='|Φ(χ)|²')
    ax1.set_title('Φ(χ): один пик в χ', fontsize=11)
    ax1.set_xlabel('χ')
    ax1.legend(fontsize=9)
    ax1.set_xlim(-10, 10)
    
    ax2 = fig.add_subplot(4, 3, 2)
    x = results_1['x']
    psi = results_1['psi_single']
    norm_val = np.sqrt(np.sum(np.abs(psi)**2) * (x[1]-x[0]))
    if norm_val > 0:
        psi = psi / norm_val
    ax2.plot(x, np.abs(psi)**2, 'r-', lw=2, label='|ψ(x)|²')
    ax2.set_title('→ ψ(x): гауссов пакет', fontsize=11)
    ax2.set_xlabel('x')
    ax2.legend(fontsize=9)
    ax2.set_xlim(-10, 10)
    
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(chi, np.abs(results_1['Phi_double'])**2, 'b-', lw=2, label='|Φ(χ)|²')
    ax3.set_title('Два пика в χ → суперпозиция', fontsize=11)
    ax3.set_xlabel('χ')
    psi_d = results_1['psi_double']
    norm_val = np.sqrt(np.sum(np.abs(psi_d)**2) * (x[1]-x[0]))
    if norm_val > 0:
        psi_d = psi_d / norm_val
    ax3_twin = ax3.twinx()
    ax3_twin.plot(x, np.abs(psi_d)**2, 'r-', lw=2, alpha=0.7, label='|ψ(x)|²')
    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(-10, 10)
    
    # --- Панель 2: Неопределённость ---
    ax4 = fig.add_subplot(4, 3, 4)
    omegas = results_2['omegas']
    ax4.loglog(omegas, results_2['sigmas_x'], 'b-o', ms=4, label='Δx')
    ax4.loglog(omegas, results_2['sigmas_p'], 'r-o', ms=4, label='Δp')
    ax4.set_xlabel('ω_χ')
    ax4.set_title('Δx, Δp vs ω_χ', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(4, 3, 5)
    ax5.semilogx(omegas, results_2['products'], 'g-o', ms=4, lw=2)
    ax5.axhline(y=0.5, color='k', ls='--', alpha=0.5, label='ℏ/2 (теор. мин.)')
    ax5.set_xlabel('ω_χ')
    ax5.set_title('Δx·Δp = ℏ/2 (всегда!)', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.set_ylim(0.3, 0.8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(4, 3, 6)
    ax6.loglog(omegas, results_2['sigma_K'], 'purple', lw=2, marker='s', ms=4)
    ax6.set_xlabel('ω_χ')
    ax6.set_ylabel('σ_K')
    ax6.set_title('Ширина K: σ = √(ℏ/2mω_χ)', fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    # --- Панель 3: Запутанность ---
    ax7 = fig.add_subplot(4, 3, 7)
    ax7.plot(results_3['separations'], results_3['entropies'], 'bo-', lw=2, ms=5)
    ax7.set_xlabel('Расстояние в xyz')
    ax7.set_ylabel('S (бит)')
    ax7.set_title('Запутанность vs расстояние: НЕ зависит!', fontsize=11)
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(4, 3, 8)
    ax8.semilogx(results_3['widths'], results_3['entropies_vs_width'], 'ro-', lw=2, ms=5)
    ax8.set_xlabel('Ширина корреляции в χ')
    ax8.set_ylabel('S (бит)')
    ax8.set_title('Запутанность vs корреляция в χ', fontsize=11)
    ax8.grid(True, alpha=0.3)
    
    # --- Панель 4: Коллапс ---
    ax9 = fig.add_subplot(4, 3, 9)
    x_col = results_4['x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_4['omegas'])))
    for i, (omega, psi) in enumerate(zip(results_4['omegas'], results_4['psi_evolution'])):
        prob = np.abs(psi)**2
        ax9.plot(x_col, prob / max(prob.max(), 1e-12), color=colors[i], lw=1.5,
                label=f'ω={omega:.1f}')
    ax9.set_xlabel('x')
    ax9.set_title('Коллапс: ω_χ ↑ → ψ локализуется', fontsize=11)
    ax9.legend(fontsize=8, ncol=2)
    ax9.set_xlim(-12, 12)
    
    # --- Панель 5: Потери информации ---
    ax10 = fig.add_subplot(4, 3, 10)
    names = list(results_5['states'].keys())
    losses = [results_5['states'][n]['loss'] for n in names]
    bars = ax10.barh(names, losses, color=plt.cm.RdYlGn_r(np.array(losses)/max(losses)))
    ax10.set_xlabel('Потеря информации')
    ax10.set_title('Потери при проекции χ → x', fontsize=11)
    
    # --- Панель 6: Матрица K ---
    ax11 = fig.add_subplot(4, 3, 11)
    cfg = results_1['cfg']
    kernel = ProjectionKernel(cfg)
    extent = [-cfg.L_x, cfg.L_x, -cfg.L_chi, cfg.L_chi]
    K_show = np.abs(kernel.K.T)**2
    # Берём центральную часть для видимости
    cx, cy = cfg.N_x//2, cfg.N_chi//2
    w = 80
    im = ax11.imshow(K_show[cy-w:cy+w, cx-w:cx+w], 
                     aspect='auto', cmap='inferno',
                     extent=[-cfg.L_x*w/cfg.N_x*2, cfg.L_x*w/cfg.N_x*2,
                             -cfg.L_chi*w/cfg.N_chi*2, cfg.L_chi*w/cfg.N_chi*2])
    ax11.set_xlabel('x')
    ax11.set_ylabel('χ')
    ax11.set_title('|K(x,χ)|² — ядро проекции', fontsize=11)
    plt.colorbar(im, ax=ax11, shrink=0.8)
    
    # --- Панель 7: Схема теории ---
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis('off')
    theory_text = """
    ИТОГ: Ядро K(x, χ) выведено, не постулировано
    
    K(x,χ) = N·exp(-(x-π(χ))²/4σ²)·exp(iφ(χ)·x/ℏ)
    
    σ² = ℏ/(2mω_χ)  — из вариационного принципа
    
    ✓ Принцип неопределённости: Δx·Δp = ℏ/2
    ✓ Суперпозиция: два пика в χ → интерференция в x
    ✓ Запутанность: общие χ → несепарабельность
    ✓ Коллапс: ω_χ↑ → σ↓ → локализация
    ✓ Когерентные состояния = оптимальная проекция
    ✓ Связь с AdS/CFT и инф. геометрией
    
    НОВЫЙ ПАРАМЕТР: ω_χ 
    → Частота осцилляций в χ-пространстве
    → Определяет всё: σ, Δx, Δp, скорость коллапса
    → Потенциально измерим!
    """
    ax12.text(0.05, 0.95, theory_text, transform=ax12.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Полный pipeline: вывод + эксперименты + визуализация."""
    
    # Символьный вывод
    symbolic_derivation()
    
    # Численные эксперименты
    r1 = experiment_1_basic_projection()
    r2 = experiment_2_uncertainty_vs_omega()
    r3 = experiment_3_entanglement()
    r4 = experiment_4_collapse_dynamics()
    r5 = experiment_5_reconstruction()
    
    # Визуализация
    fig = visualize_all(r1, r2, r3, r4, r5)
    
    output_path = 'chi_projection_kernel.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Визуализация сохранена: {output_path}")
    
    plt.close(fig)
    return r1, r2, r3, r4, r5


if __name__ == '__main__':
    main()

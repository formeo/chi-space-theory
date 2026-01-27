"""
EPR + Ψ-Field: No-Signaling Test
=================================

ВОПРОС: Можно ли использовать пси-поле для сверхсветовой связи?

SETUP:
    1. Создаём EPR пару (запутанные спины)
    2. Алиса и Боб разлетаются на большое расстояние
    3. Алиса использует χ-детектор (читает спин БЕЗ коллапса)
    4. Проверяем: изменилась ли статистика у Боба?

СТАНДАРТНАЯ КМ:
    - Измерение Алисы → коллапс → корреляция с Бобом
    - НО: Боб не может узнать ЧТО измерила Алиса без классического канала
    - No-signaling: P(Bob) не зависит от действий Алисы

ПСИ-ПОЛЕ (гипотеза):
    - χ-чтение БЕЗ коллапса
    - Если χ нелокально → Боб может "почувствовать" действия Алисы?
    - Это нарушило бы no-signaling → FTL связь!

Author: Roman
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import seaborn as sns

sns.set_style("whitegrid")


class SpinState(Enum):
    """Спиновые состояния."""
    UP = 1
    DOWN = -1


@dataclass
class EPRPair:
    """
    EPR пара (запутанное состояние двух спинов).
    
    |Ψ⟩ = (|↑↓⟩ - |↓↑⟩) / √2  (синглетное состояние)
    
    Свойства:
    - Полный спин = 0
    - Измерение одного МГНОВЕННО определяет другой
    - Но корреляции не позволяют передать информацию (no-signaling)
    """
    # Амплитуды: |↑↓⟩ и |↓↑⟩
    alpha: complex = 1/np.sqrt(2)   # амплитуда |↑↓⟩ (Алиса↑, Боб↓)
    beta: complex = -1/np.sqrt(2)   # амплитуда |↓↑⟩ (Алиса↓, Боб↑)
    
    collapsed: bool = False
    alice_state: SpinState | None = None
    bob_state: SpinState | None = None
    
    # Пси-поле: χ значения (скрытые переменные)
    chi_alice: float | None = None
    chi_bob: float | None = None
    
    def __post_init__(self):
        """Инициализация χ-поля."""
        # χ кодирует "предопределённое" состояние
        # В стандартной КМ это не имеет смысла (нет скрытых переменных)
        # В пси-поле: χ существует ДО измерения
        self._init_chi_field()
    
    def _init_chi_field(self):
        """
        Инициализация пси-поля.
        
        ПОСТУЛАТ: χ определяет "реальное" состояние до измерения.
        Для EPR пары: χ_alice + χ_bob = 0 (сохранение спина)
        """
        # Случайно выбираем "скрытое" состояние
        if np.random.random() < np.abs(self.alpha)**2:
            # Реализация |↑↓⟩
            self.chi_alice = +1.0
            self.chi_bob = -1.0
        else:
            # Реализация |↓↑⟩
            self.chi_alice = -1.0
            self.chi_bob = +1.0
    
    def get_probabilities(self) -> tuple[float, float]:
        """Вероятности P(↑↓) и P(↓↑)."""
        p_up_down = np.abs(self.alpha)**2
        p_down_up = np.abs(self.beta)**2
        return p_up_down, p_down_up
    
    @property
    def is_entangled(self) -> bool:
        """Проверка запутанности."""
        return not self.collapsed


class StandardQMMeasurement:
    """
    Стандартное квантовое измерение.
    
    Измерение → Коллапс → Корреляция
    """
    
    @staticmethod
    def measure_alice(pair: EPRPair) -> SpinState:
        """
        Алиса измеряет свой спин (стандартная КМ).
        
        Результат: коллапс волновой функции!
        """
        if pair.collapsed:
            return pair.alice_state
        
        p_up, p_down = pair.get_probabilities()
        
        if np.random.random() < p_up:
            # Коллапс в |↑↓⟩
            pair.alice_state = SpinState.UP
            pair.bob_state = SpinState.DOWN
        else:
            # Коллапс в |↓↑⟩
            pair.alice_state = SpinState.DOWN
            pair.bob_state = SpinState.UP
        
        pair.collapsed = True
        pair.alpha = 1.0 if pair.alice_state == SpinState.UP else 0.0
        pair.beta = 0.0 if pair.alice_state == SpinState.UP else 1.0
        
        return pair.alice_state
    
    @staticmethod
    def measure_bob(pair: EPRPair) -> SpinState:
        """Боб измеряет свой спин."""
        if pair.collapsed:
            return pair.bob_state
        
        # Если Алиса ещё не измеряла — Боб коллапсирует первым
        p_up, p_down = pair.get_probabilities()
        
        # P(Bob=↑) = P(↓↑) = |beta|²
        if np.random.random() < p_down:
            pair.bob_state = SpinState.UP
            pair.alice_state = SpinState.DOWN
        else:
            pair.bob_state = SpinState.DOWN
            pair.alice_state = SpinState.UP
        
        pair.collapsed = True
        return pair.bob_state


class PsiFieldDetector:
    """
    Ψ-Field χ-детектор.
    
    ПОСТУЛАТ: Может читать χ БЕЗ коллапса волновой функции.
    
    Если это возможно:
    - Алиса узнаёт состояние без разрушения запутанности
    - Запутанность сохраняется
    - Возможна ли передача информации Бобу?
    """
    
    def __init__(self, fidelity: float = 0.95):
        """
        Args:
            fidelity: Точность χ-детектора (0.5 = случайный, 1.0 = идеальный)
        """
        self.fidelity = fidelity
    
    def read_chi_alice(self, pair: EPRPair, collapse: bool = False) -> float:
        """
        Алиса читает χ своей частицы.
        
        Args:
            pair: EPR пара
            collapse: Если True — стандартное измерение с коллапсом
                     Если False — пси-поле чтение БЕЗ коллапса (гипотеза!)
        
        Returns:
            χ значение (+1 или -1)
        """
        true_chi = pair.chi_alice
        
        # Добавляем шум детектора
        if np.random.random() < self.fidelity:
            measured_chi = true_chi
        else:
            measured_chi = -true_chi  # Ошибка
        
        if collapse:
            # Стандартная КМ: коллапс при измерении
            pair.collapsed = True
            pair.alice_state = SpinState.UP if measured_chi > 0 else SpinState.DOWN
            pair.bob_state = SpinState.DOWN if measured_chi > 0 else SpinState.UP
        
        # Если collapse=False — ПСИ-ПОЛЕ: читаем без коллапса!
        # pair.collapsed остаётся False
        # Это "магия" пси-поля
        
        return measured_chi
    
    def read_chi_bob(self, pair: EPRPair) -> float:
        """Боб читает χ своей частицы."""
        true_chi = pair.chi_bob
        
        if np.random.random() < self.fidelity:
            return true_chi
        else:
            return -true_chi


class NoSignalingExperiment:
    """
    Эксперимент на проверку no-signaling теоремы.
    
    ПРОТОКОЛ:
    1. Создаём N EPR пар
    2. Половина: Алиса читает χ (пси-поле, без коллапса)
    3. Половина: Алиса НЕ читает (контроль)
    4. Боб измеряет ВСЕ свои частицы (стандартно)
    5. Сравниваем статистику Боба в двух группах
    
    ЕСЛИ статистика РАЗНАЯ → можно сигналить через пси-поле!
    ЕСЛИ статистика ОДИНАКОВАЯ → no-signaling сохраняется
    """
    
    def __init__(self, n_pairs: int = 5000, chi_fidelity: float = 0.95):
        self.n_pairs = n_pairs
        self.psi_detector = PsiFieldDetector(fidelity=chi_fidelity)
        
        self.results = {
            'control': {'bob_ups': 0, 'bob_downs': 0},
            'psi_field': {'bob_ups': 0, 'bob_downs': 0},
            'psi_field_corr': {'same': 0, 'diff': 0},  # Корреляция χ_alice ↔ bob_measurement
        }
    
    def run(self) -> dict:
        """Запуск эксперимента."""
        print("="*70)
        print("EPR + Ψ-FIELD: NO-SIGNALING TEST")
        print("="*70)
        print(f"\nN pairs: {self.n_pairs}")
        print(f"χ-detector fidelity: {self.psi_detector.fidelity:.1%}")
        print("\n" + "-"*70)
        
        # Группа 1: Контроль (Алиса ничего не делает)
        print("\n[1] CONTROL: Alice does nothing...")
        for _ in range(self.n_pairs // 2):
            pair = EPRPair()
            
            # Алиса НЕ измеряет
            # Боб измеряет стандартно
            bob_result = StandardQMMeasurement.measure_bob(pair)
            
            if bob_result == SpinState.UP:
                self.results['control']['bob_ups'] += 1
            else:
                self.results['control']['bob_downs'] += 1
        
        # Группа 2: Пси-поле (Алиса читает χ без коллапса)
        print("[2] PSI-FIELD: Alice reads χ (no collapse)...")
        for _ in range(self.n_pairs // 2):
            pair = EPRPair()
            
            # Алиса читает χ через пси-поле (БЕЗ коллапса!)
            alice_chi = self.psi_detector.read_chi_alice(pair, collapse=False)
            
            # Запутанность должна сохраниться!
            assert pair.is_entangled, "Psi-field should not collapse!"
            
            # Боб измеряет стандартно
            bob_result = StandardQMMeasurement.measure_bob(pair)
            
            if bob_result == SpinState.UP:
                self.results['psi_field']['bob_ups'] += 1
            else:
                self.results['psi_field']['bob_downs'] += 1
            
            # Корреляция: совпал ли χ_alice с bob_measurement?
            # Если χ_alice = +1 (↑), то Bob должен быть ↓
            expected_bob = SpinState.DOWN if alice_chi > 0 else SpinState.UP
            if bob_result == expected_bob:
                self.results['psi_field_corr']['same'] += 1
            else:
                self.results['psi_field_corr']['diff'] += 1
        
        return self.results
    
    def analyze(self) -> dict:
        """Анализ результатов."""
        ctrl = self.results['control']
        psi = self.results['psi_field']
        corr = self.results['psi_field_corr']
        
        n_ctrl = ctrl['bob_ups'] + ctrl['bob_downs']
        n_psi = psi['bob_ups'] + psi['bob_downs']
        
        # Вероятности P(Bob=↑)
        p_ctrl = ctrl['bob_ups'] / n_ctrl
        p_psi = psi['bob_ups'] / n_psi
        
        # Корреляция χ_alice ↔ bob_measurement
        correlation = corr['same'] / (corr['same'] + corr['diff'])
        
        # Статистический тест (z-test для пропорций)
        p_pooled = (ctrl['bob_ups'] + psi['bob_ups']) / (n_ctrl + n_psi)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_ctrl + 1/n_psi))
        z_score = (p_ctrl - p_psi) / se if se > 0 else 0
        
        analysis = {
            'p_bob_up_control': p_ctrl,
            'p_bob_up_psi_field': p_psi,
            'difference': p_psi - p_ctrl,
            'z_score': z_score,
            'chi_bob_correlation': correlation,
            'no_signaling_holds': abs(z_score) < 2.0,  # 95% confidence
        }
        
        return analysis
    
    def print_results(self, analysis: dict):
        """Вывод результатов."""
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        ctrl = self.results['control']
        psi = self.results['psi_field']
        
        print(f"\n{'Group':<20} {'Bob ↑':<12} {'Bob ↓':<12} {'P(↑)':<12}")
        print("-"*56)
        print(f"{'Control':<20} {ctrl['bob_ups']:<12} {ctrl['bob_downs']:<12} {analysis['p_bob_up_control']:.4f}")
        print(f"{'Ψ-Field':<20} {psi['bob_ups']:<12} {psi['bob_downs']:<12} {analysis['p_bob_up_psi_field']:.4f}")
        
        print(f"\n{'Difference:':<20} {analysis['difference']:+.4f}")
        print(f"{'Z-score:':<20} {analysis['z_score']:.2f}")
        print(f"{'χ↔Bob correlation:':<20} {analysis['chi_bob_correlation']:.2%}")
        
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        
        if analysis['no_signaling_holds']:
            print("""
    ✓ NO-SIGNALING HOLDS
    
    Статистика Боба ОДИНАКОВА независимо от действий Алисы.
    Даже если Алиса читает χ через пси-поле:
    → Боб НЕ может узнать об этом
    → Передача информации НЕВОЗМОЖНА
    → FTL связь не работает
    
    ВЫВОД: Пси-поле (если существует) НЕ нарушает причинность.
            """)
        else:
            print("""
    ⚠️  NO-SIGNALING VIOLATION DETECTED!
    
    Статистика Боба ЗАВИСИТ от действий Алисы!
    → Можно передавать информацию мгновенно
    → FTL связь возможна!
    → Нарушение причинности!
    
    ВЫВОД: Либо ошибка в симуляции, либо новая физика!
            """)
        
        print(f"\nКорреляция χ_alice ↔ Bob: {analysis['chi_bob_correlation']:.1%}")
        print("(Насколько χ-чтение Алисы предсказывает результат Боба)")
        
        if analysis['chi_bob_correlation'] > 0.9:
            print("\n→ Высокая корреляция! χ действительно 'знает' состояние.")
            print("   Но это не помогает сигналить (Боб не видит χ напрямую).")
    
    def plot_results(self, analysis: dict) -> plt.Figure:
        """Визуализация результатов."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("EPR + Ψ-Field: No-Signaling Test", fontsize=14, fontweight='bold')
        
        # 1. Сравнение статистики Боба
        ax1 = axes[0]
        groups = ['Control', 'Ψ-Field']
        bob_ups = [self.results['control']['bob_ups'], 
                   self.results['psi_field']['bob_ups']]
        bob_downs = [self.results['control']['bob_downs'],
                     self.results['psi_field']['bob_downs']]
        
        x = np.arange(len(groups))
        width = 0.35
        
        ax1.bar(x - width/2, bob_ups, width, label='Bob ↑', color='steelblue')
        ax1.bar(x + width/2, bob_downs, width, label='Bob ↓', color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups)
        ax1.set_ylabel('Count')
        ax1.set_title("Bob's Measurements\n(should be identical)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. P(Bob=↑) сравнение
        ax2 = axes[1]
        probs = [analysis['p_bob_up_control'], analysis['p_bob_up_psi_field']]
        colors = ['steelblue', 'mediumseagreen']
        bars = ax2.bar(groups, probs, color=colors, edgecolor='black')
        ax2.axhline(y=0.5, color='red', linestyle='--', label='Expected (0.5)')
        ax2.set_ylabel('P(Bob = ↑)')
        ax2.set_title(f"Probability Comparison\nΔ = {analysis['difference']:+.4f}")
        ax2.set_ylim([0.4, 0.6])
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, prob in zip(bars, probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=11)
        
        # 3. χ-Bob корреляция
        ax3 = axes[2]
        corr = self.results['psi_field_corr']
        labels = ['χ predicts\ncorrectly', 'χ predicts\nwrongly']
        sizes = [corr['same'], corr['diff']]
        colors_pie = ['mediumseagreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0))
        ax3.set_title(f"χ-Measurement Correlation\n(Fidelity: {self.psi_detector.fidelity:.0%})")
        
        plt.tight_layout()
        return fig


class PsiFieldTeleportationTest:
    """
    Тест: Можно ли использовать пси-поле для телепортации?
    
    ИДЕЯ:
    1. Алиса хочет передать бит информации Бобу
    2. У них есть запутанная пара
    3. Алиса кодирует бит через χ-манипуляцию (???)
    4. Боб декодирует через своё измерение
    
    ПРОБЛЕМА: Как Алиса может ИЗМЕНИТЬ χ?
    """
    
    def __init__(self, n_trials: int = 1000):
        self.n_trials = n_trials
        self.psi_detector = PsiFieldDetector(fidelity=0.99)
    
    def test_information_transfer(self) -> dict:
        """
        Попытка передать информацию через пси-поле.
        
        Протокол:
        - Алиса хочет передать бит b ∈ {0, 1}
        - Если b=0: Алиса НЕ читает χ
        - Если b=1: Алиса читает χ (пси-поле)
        - Боб пытается угадать b по своему измерению
        """
        print("\n" + "="*70)
        print("PSI-FIELD INFORMATION TRANSFER TEST")
        print("="*70)
        
        correct_guesses = 0
        
        for i in range(self.n_trials):
            # Алиса выбирает бит для передачи
            alice_bit = np.random.randint(0, 2)
            
            # Создаём EPR пару
            pair = EPRPair()
            
            if alice_bit == 1:
                # Алиса читает χ (пытается "пометить" пару)
                _ = self.psi_detector.read_chi_alice(pair, collapse=False)
            
            # Боб измеряет и пытается угадать бит Алисы
            bob_result = StandardQMMeasurement.measure_bob(pair)
            
            # Стратегия Боба: угадать на основе результата
            # (любая стратегия должна давать ~50%)
            bob_guess = 1 if bob_result == SpinState.UP else 0
            
            if bob_guess == alice_bit:
                correct_guesses += 1
        
        accuracy = correct_guesses / self.n_trials
        
        print(f"\nTrials: {self.n_trials}")
        print(f"Bob's accuracy: {accuracy:.2%}")
        print(f"Expected (random): 50%")
        
        if abs(accuracy - 0.5) < 0.05:
            print("\n→ Bob cannot detect Alice's actions")
            print("→ Information transfer FAILED")
            print("→ No-signaling preserved!")
        else:
            print("\n⚠️ Anomaly detected!")
            print(f"→ Deviation from 50%: {abs(accuracy - 0.5):.1%}")
        
        return {'accuracy': accuracy, 'n_trials': self.n_trials}


def main():
    """Запуск всех экспериментов."""
    
    # Эксперимент 1: No-Signaling тест
    print("\n" + "█"*70)
    print("█" + " "*25 + "EXPERIMENT 1" + " "*25 + "█")
    print("█"*70)
    
    exp1 = NoSignalingExperiment(n_pairs=6000, chi_fidelity=0.95)
    exp1.run()
    analysis = exp1.analyze()
    exp1.print_results(analysis)
    fig1 = exp1.plot_results(analysis)
    
    # Эксперимент 2: Попытка передачи информации
    print("\n" + "█"*70)
    print("█" + " "*25 + "EXPERIMENT 2" + " "*25 + "█")
    print("█"*70)
    
    exp2 = PsiFieldTeleportationTest(n_trials=2000)
    exp2.test_information_transfer()
    
    # Итоговый вывод
    print("\n" + "="*70)
    print("FINAL CONCLUSIONS")
    print("="*70)
    print("""
    1. NO-SIGNALING ТЕОРЕМА СОХРАНЯЕТСЯ
       Даже с "волшебным" χ-детектором Боб не видит действий Алисы.
       
    2. ПРИЧИНА: χ-чтение не меняет СТАТИСТИКУ
       - Алиса узнаёт χ → знает что получит Боб
       - Но Боб получает СЛУЧАЙНЫЙ результат (50/50)
       - Без классического канала Боб не знает что "предсказала" Алиса
       
    3. ДЛЯ ТЕЛЕПОРТАЦИИ НУЖНО БОЛЬШЕ:
       - Не просто ЧИТАТЬ χ, а ИЗМЕНЯТЬ его
       - Или: χ должно быть НЕЛОКАЛЬНЫМ (одно χ на двоих)
       - Или: нужен механизм синхронизации χ-полей
       
    4. СЛЕДУЮЩИЙ ШАГ:
       Моделировать НЕЛОКАЛЬНОЕ пси-поле:
       χ_total = f(χ_alice, χ_bob) с мгновенной связью
    """)
    
    return fig1, analysis


if __name__ == "__main__":
    fig, analysis = main()
    
    fig.savefig('/mnt/user-data/outputs/epr_psi_field_test.png',
                dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Plot saved: outputs/epr_psi_field_test.png")

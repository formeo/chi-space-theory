"""
Нелокальное Ψ-Поле: Попытка телепортации
=========================================

ПРОБЛЕМА предыдущего эксперимента:
    - χ-чтение не меняет статистику Боба
    - Нужно ВЛИЯТЬ на χ, а не просто читать

НОВАЯ ГИПОТЕЗА: Нелокальное пси-поле
    - χ — это ОБЩЕЕ поле для запутанных частиц
    - Изменение χ у Алисы МГНОВЕННО меняет χ у Боба
    - Это как "квантовый телефон"

МАТЕМАТИЧЕСКИ:
    |Ψ⟩ ⊗ |χ⟩  где |χ⟩ — нелокальное пси-поле
    
    Операция Алисы: U_A ⊗ V_χ
    Где V_χ действует на ОБЩЕЕ χ → Боб "чувствует" изменение

ОПАСНОСТЬ:
    Если это работает → нарушение причинности!
    
Author: Roman
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple
import seaborn as sns

sns.set_style("whitegrid")


@dataclass
class NonlocalPsiField:
    """
    Нелокальное пси-поле.
    
    ПОСТУЛАТ: χ — это общее "скрытое измерение" для запутанных частиц.
    Действия над χ_local влияют на χ_global мгновенно.
    
    Модель:
        χ_global ∈ [-1, +1] — состояние поля
        χ_alice, χ_bob — локальные "окна" в поле
        
    Связь:
        χ_bob = f(χ_global, action_alice)
    """
    
    # Глобальное состояние пси-поля
    chi_global: float = 0.0
    
    # Локальные состояния (проекции глобального)
    chi_alice: float = 0.0
    chi_bob: float = 0.0
    
    # Параметр связи (насколько сильно действия Алисы влияют на Боба)
    coupling: float = 0.8
    
    # История изменений (для отладки)
    history: List[dict] = field(default_factory=list)
    
    def initialize_entangled(self):
        """Инициализация для запутанной пары."""
        # Случайное начальное состояние
        self.chi_global = np.random.choice([-1.0, +1.0])
        
        # Антикорреляция (как в EPR)
        self.chi_alice = self.chi_global
        self.chi_bob = -self.chi_global
        
        self.history.append({
            'action': 'init',
            'chi_global': self.chi_global,
            'chi_alice': self.chi_alice,
            'chi_bob': self.chi_bob,
        })
    
    def alice_flip(self):
        """
        Алиса "переворачивает" своё χ.
        
        КЛЮЧЕВАЯ ГИПОТЕЗА: Это влияет на χ_bob мгновенно!
        """
        self.chi_alice = -self.chi_alice
        
        # Нелокальное влияние на Боба
        # Модель: частичная корреляция
        delta = self.coupling * (-2 * self.chi_bob)  # Сигнал к перевороту
        noise = np.random.normal(0, 0.1)
        
        # Боб "чувствует" переворот Алисы
        self.chi_bob = np.clip(self.chi_bob + delta + noise, -1, 1)
        
        # Обновляем глобальное
        self.chi_global = -self.chi_global
        
        self.history.append({
            'action': 'alice_flip',
            'chi_global': self.chi_global,
            'chi_alice': self.chi_alice,
            'chi_bob': self.chi_bob,
        })
    
    def alice_read(self) -> float:
        """Алиса читает своё χ (без изменения)."""
        return self.chi_alice
    
    def bob_read(self) -> float:
        """Боб читает своё χ."""
        return self.chi_bob


@dataclass  
class QuantumState:
    """Квантовое состояние спина (для сравнения с классической КМ)."""
    # |ψ⟩ = α|↑⟩ + β|↓⟩
    alpha: complex = 1/np.sqrt(2)
    beta: complex = 1/np.sqrt(2)
    
    def measure(self) -> int:
        """Измерение → коллапс."""
        p_up = np.abs(self.alpha)**2
        result = 1 if np.random.random() < p_up else -1
        
        # Коллапс
        if result == 1:
            self.alpha, self.beta = 1.0, 0.0
        else:
            self.alpha, self.beta = 0.0, 1.0
        
        return result


class NonlocalTeleportationExperiment:
    """
    Эксперимент: Телепортация через нелокальное пси-поле.
    
    ПРОТОКОЛ:
    1. Создаём запутанную пару + нелокальное χ-поле
    2. Алиса кодирует бит: 0 → ничего, 1 → flip χ
    3. Боб читает своё χ и декодирует бит
    
    ЕСЛИ работает → мгновенная передача информации!
    """
    
    def __init__(self, coupling: float = 0.9):
        self.coupling = coupling
        self.results = []
    
    def run_single_trial(self, alice_bit: int) -> Tuple[int, int]:
        """
        Один раунд телепортации.
        
        Returns:
            (alice_bit, bob_decoded_bit)
        """
        # Создаём пси-поле
        psi = NonlocalPsiField(coupling=self.coupling)
        psi.initialize_entangled()
        
        # Боб запоминает начальное χ
        bob_initial = psi.bob_read()
        
        # Алиса кодирует
        if alice_bit == 1:
            psi.alice_flip()
        
        # Боб читает и декодирует
        bob_final = psi.bob_read()
        
        # Стратегия декодирования: изменилось ли χ?
        if np.sign(bob_final) != np.sign(bob_initial):
            bob_decoded = 1  # Было изменение → Алиса послала 1
        else:
            bob_decoded = 0  # Не изменилось → Алиса послала 0
        
        return alice_bit, bob_decoded
    
    def run_experiment(self, n_trials: int = 2000) -> dict:
        """Запуск эксперимента."""
        print("="*70)
        print("NONLOCAL Ψ-FIELD TELEPORTATION TEST")
        print("="*70)
        print(f"\nCoupling strength: {self.coupling}")
        print(f"Trials: {n_trials}")
        
        correct = 0
        confusion = {'00': 0, '01': 0, '10': 0, '11': 0}
        
        for _ in range(n_trials):
            alice_bit = np.random.randint(0, 2)
            _, bob_decoded = self.run_single_trial(alice_bit)
            
            key = f"{alice_bit}{bob_decoded}"
            confusion[key] += 1
            
            if alice_bit == bob_decoded:
                correct += 1
        
        accuracy = correct / n_trials
        
        results = {
            'accuracy': accuracy,
            'n_trials': n_trials,
            'coupling': self.coupling,
            'confusion': confusion,
        }
        
        self.results.append(results)
        return results
    
    def print_results(self, results: dict):
        """Вывод результатов."""
        print(f"\n{'='*70}")
        print("RESULTS")
        print("="*70)
        
        print(f"\nAccuracy: {results['accuracy']:.2%}")
        print(f"Expected random: 50%")
        print(f"Improvement: {(results['accuracy'] - 0.5) * 100:+.1f}%")
        
        print("\nConfusion Matrix:")
        print("                 Bob decoded")
        print("              |   0   |   1   |")
        print("-" * 35)
        c = results['confusion']
        print(f"Alice sent 0  | {c['00']:5} | {c['01']:5} |")
        print(f"Alice sent 1  | {c['10']:5} | {c['11']:5} |")
        
        if results['accuracy'] > 0.6:
            print("\n" + "🔥"*20)
            print("⚠️  INFORMATION TRANSFER DETECTED!")
            print("🔥"*20)
            print(f"""
    Боб декодирует бит Алисы с точностью {results['accuracy']:.1%}!
    Это ВЫШЕ случайных 50%!
    
    ЕСЛИ ЭТО РЕАЛЬНО:
    → Нелокальное пси-поле существует
    → Мгновенная связь возможна
    → Нарушение причинности!
    
    НО СКОРЕЕ ВСЕГО:
    → Это артефакт модели
    → В реальности no-signaling должен работать
            """)
        else:
            print("\n✓ No significant information transfer")
            print("  No-signaling appears to hold")


class CouplingStrengthScan:
    """Сканирование по силе связи пси-поля."""
    
    def __init__(self):
        self.results = []
    
    def scan(self, couplings: List[float], n_trials: int = 1000):
        """Сканирование по разным значениям coupling."""
        print("\n" + "="*70)
        print("COUPLING STRENGTH SCAN")
        print("="*70)
        
        for coupling in couplings:
            exp = NonlocalTeleportationExperiment(coupling=coupling)
            result = exp.run_experiment(n_trials)
            self.results.append(result)
            print(f"  coupling={coupling:.2f} → accuracy={result['accuracy']:.2%}")
        
        return self.results
    
    def plot(self) -> plt.Figure:
        """График зависимости точности от силы связи."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        couplings = [r['coupling'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        
        ax.plot(couplings, accuracies, 'o-', markersize=10, linewidth=2,
               color='steelblue', label='Measured accuracy')
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
                  label='Random guess (50%)')
        ax.axhline(y=1.0, color='green', linestyle=':', linewidth=2,
                  label='Perfect transfer (100%)')
        
        ax.fill_between(couplings, 0.5, accuracies, alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Ψ-Field Coupling Strength', fontsize=12)
        ax.set_ylabel('Bob\'s Decoding Accuracy', fontsize=12)
        ax.set_title('Nonlocal Ψ-Field: Information Transfer vs Coupling', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0.4, 1.05])
        ax.set_xlim([0, 1])
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        # Добавляем аннотации
        for c, a in zip(couplings, accuracies):
            if a > 0.55:
                ax.annotate(f'{a:.0%}', (c, a), textcoords="offset points",
                           xytext=(0, 10), ha='center')
        
        return fig


def theoretical_analysis():
    """Теоретический анализ результатов."""
    print("\n" + "="*70)
    print("THEORETICAL ANALYSIS")
    print("="*70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ЧТО ПОКАЗАЛА СИМУЛЯЦИЯ                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. С НЕЛОКАЛЬНЫМ ПСИ-ПОЛЕМ передача информации ВОЗМОЖНА       │
    │     (в рамках нашей модели)                                    │
    │                                                                 │
    │  2. Точность зависит от силы связи (coupling):                 │
    │     • coupling = 0   → 50% (случайно)                          │
    │     • coupling = 0.5 → ~75%                                    │
    │     • coupling = 1.0 → ~95%                                    │
    │                                                                 │
    │  3. ЭТО НАРУШАЕТ NO-SIGNALING!                                 │
    │     → В реальной физике это ЗАПРЕЩЕНО                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ПОЧЕМУ ЭТО НЕ РЕАЛЬНО                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ПРОБЛЕМА 1: Причинность                                       │
    │  • Если Алиса влияет на Боба мгновенно                        │
    │  • В разных системах отсчёта порядок событий РАЗНЫЙ           │
    │  • Возникают парадоксы (убийство дедушки и т.п.)              │
    │                                                                 │
    │  ПРОБЛЕМА 2: Энергия                                           │
    │  • Передача информации требует энергии                         │
    │  • Откуда берётся энергия для нелокального влияния?           │
    │                                                                 │
    │  ПРОБЛЕМА 3: Все эксперименты                                  │
    │  • Тысячи тестов Bell inequalities                            │
    │  • Все подтверждают no-signaling                               │
    │  • Никаких намёков на нелокальное пси-поле                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ВОЗМОЖНЫЕ ЛАЗЕЙКИ                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ИДЕЯ 1: Пси-поле работает только на планковских масштабах    │
    │  • L_p ~ 10^-35 м                                              │
    │  • Эффекты слишком малы для детекции                          │
    │  • Но могут накапливаться?                                    │
    │                                                                 │
    │  ИДЕЯ 2: Пси-поле требует особых условий                      │
    │  • Сверхнизкие температуры?                                    │
    │  • Определённые материалы?                                     │
    │  • Особая геометрия?                                          │
    │                                                                 │
    │  ИДЕЯ 3: Пси-поле ≠ передача информации                       │
    │  • Может переносить "что-то другое"                           │
    │  • Не нарушает no-signaling напрямую                          │
    │  • Но создаёт предпосылки для телепортации                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Главная функция."""
    
    # Эксперимент 1: Фиксированная связь
    print("\n" + "█"*70)
    print("█" + " "*20 + "EXPERIMENT 1: Fixed Coupling" + " "*15 + "█")
    print("█"*70)
    
    exp1 = NonlocalTeleportationExperiment(coupling=0.8)
    results1 = exp1.run_experiment(n_trials=3000)
    exp1.print_results(results1)
    
    # Эксперимент 2: Сканирование по coupling
    print("\n" + "█"*70)
    print("█" + " "*20 + "EXPERIMENT 2: Coupling Scan" + " "*16 + "█")
    print("█"*70)
    
    scanner = CouplingStrengthScan()
    scanner.scan(
        couplings=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0],
        n_trials=1500
    )
    fig = scanner.plot()
    
    # Теоретический анализ
    theoretical_analysis()
    
    # Итоговые выводы
    print("\n" + "="*70)
    print("NEXT STEPS FOR PSI-FIELD THEORY")
    print("="*70)
    print("""
    1. МАТЕМАТИКА:
       • Построить лагранжиан для нелокального пси-поля
       • Проверить совместимость с лоренц-инвариантностью
       • Исследовать предел слабой связи (coupling → 0)
    
    2. ФИЗИКА:
       • Найти механизм, который делает пси-поле возможным
       • Связь с тёмной материей/энергией?
       • Связь с квантовой гравитацией?
    
    3. ЭКСПЕРИМЕНТ:
       • Придумать тест, который отличит пси-поле от стандартной КМ
       • Precision measurements на запутанных парах
       • Поиск аномалий в корреляциях
    
    4. ТЕЛЕПОРТАЦИЯ:
       • Если пси-поле существует: как его использовать?
       • Масштабирование от частиц к макрообъектам
       • Инженерные проблемы
    """)
    
    return fig


if __name__ == "__main__":
    fig = main()
    
    fig.savefig('/mnt/user-data/outputs/nonlocal_psi_teleportation.png',
                dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Plot saved: outputs/nonlocal_psi_teleportation.png")

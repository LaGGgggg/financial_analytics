# Factor-Based Investment Strategy Analyzer

Анализатор классических инвестиционных стратегий на российском рынке акций.
В проекте реализованы пять стратегий (Passive, Momentum, Value, Growth, Quality). 
Выходные данные: графики, реализующие сравнение стратегий:
1. График накопленной доходности на общем периоде 
 2. График годовых доходностей
 3. График Risk/Return
 4. График превышения доходности над Passive
 5. График turnover портфеля
 6. График среднего пересечения портфелей стратегий




## Обзор проекта
Проект реализован на Python с использованием библиотек `pandas`, `numpy` и `matplotlib`.

Проект позволяет:
- Загружать исторические данные о ценах акций и фундаментальные показатели
- Проводить сравнение пяти инвестиционных стратегий
- Сравнивать стратегии по доходности, риску, оборачиваемости и составу портфелей
- Визуализировать результаты через набор стандартизированных графиков

---
# Запуск проекта

1. Clone this repository: `git clone https://github.com/LaGGgggg/financial_analytics`
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
3. Download and copy history data (or you can just wait and the project parse it automatically):

    Download all .json files from the [releases page](https://github.com/LaGGgggg/financial_analytics/releases/tag/latest)
    and save them into a `data` directory inside the project (create if not exists)
4. Run the project: `uv run main.py`


# Структура проекта

```text

project/
├── analyzers/
│   ├── __init__.py
│   ├── base.py                    # Базовый класс StrategyAnalyzerBase
│   ├── passive_strategy.py        # Пассивная стратегия PassiveStrategyAnalyzer
│   ├── momentum_strategy.py       # Momentum стратегия MomentumStrategyAnalyzer
│   ├── value_strategy.py          # Value стратегия ValueStrategyAnalyzer
│   ├── growth_strategy.py         # Growth стратегия GrowthStrategyAnalyzer
│   └── quality_strategy.py        # Quality стратегия QualityStrategyAnalyzer
├── plotter/
│   ├── __init__.py
│   └── comparison_plotter.py      # Построение графиков StrategyComparisonPlotter
├── data/
│   ├── prices.csv                 # Исторические цены акций
│   └── fundamentals.csv           # Фундаментальные показатели
├── main.py                        # Точка входа для запуска анализа
├── requirements.txt               # Зависимости
└── README.md                      # Документация проекта
```

## Реализованные стратегии

### 1. Passive (Пассивная стратегия)
**Базовый ориентир для сравнения.** Все бумаги включаются в портфель с равными весами. Не использует фундаментальные показатели, мультипликаторы или прошлую доходность.

## 2. Momentum-стратегия

**Momentum-стратегия** основана на предположении, что бумаги, которые хорошо росли в прошлом периоде, могут продолжить показывать относительно сильную динамику в следующем периоде.

## 3. Value-стратегия

**Value-стратегия** выбирает бумаги, которые выглядят недооценёнными по фундаментальным мультипликаторам.

## 4. Growth-стратегия

**Growth-стратегия** выбирает компании, у которых быстрее растут основные финансовые показатели.

## 5. Quality-стратегия

**Quality-стратегия** выбирает компании с более качественными финансовыми характеристиками: высокой рентабельностью и умеренной долговой нагрузкой.

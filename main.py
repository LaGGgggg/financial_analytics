import matplotlib.pyplot as plt
import pandas as pd

from analyzers import (
    GrowthStrategyAnalyzer,
    MomentumStrategyAnalyzer,
    PassiveStrategyAnalyzer,
    QualityStrategyAnalyzer,
    ValueStrategyAnalyzer,
)
from compare import StrategyComparisonPlotter
from parsers import IssParser, SmartLabParser
from settings import TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH


def main() -> None:

    IssParser().parse_and_save()
    SmartLabParser().parse_and_save(
        pd.read_json(TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH)['SECID'].dropna().drop_duplicates().tolist()
    )

    comparison = StrategyComparisonPlotter(
        analyzers=[
            PassiveStrategyAnalyzer(),
            MomentumStrategyAnalyzer(),
            ValueStrategyAnalyzer(),
            GrowthStrategyAnalyzer(),
            QualityStrategyAnalyzer(),
        ],
    )

    plt.ion()

    comparison.plot_wealth_index()
    comparison.plot_annual_returns()
    comparison.plot_risk_return()
    comparison.plot_excess_returns()
    comparison.plot_turnover()
    comparison.plot_strategy_overlap_summary()

    comparison.plot_strategy_holdings_table(PassiveStrategyAnalyzer.STRATEGY_NAME, 2024)
    comparison.plot_strategy_holdings_table(MomentumStrategyAnalyzer.STRATEGY_NAME, 2024)
    comparison.plot_strategy_holdings_table(ValueStrategyAnalyzer.STRATEGY_NAME, 2024)
    comparison.plot_strategy_holdings_table(GrowthStrategyAnalyzer.STRATEGY_NAME, 2024)
    comparison.plot_strategy_holdings_table(QualityStrategyAnalyzer.STRATEGY_NAME, 2024)

    plt.show(block=True)

if __name__ == '__main__':
    main()

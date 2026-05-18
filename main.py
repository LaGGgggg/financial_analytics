import pandas as pd
import matplotlib.pyplot as plt

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
    plt.ion()
    comparison = StrategyComparisonPlotter(
        analyzers=[
            PassiveStrategyAnalyzer(),
            MomentumStrategyAnalyzer(),
            ValueStrategyAnalyzer(),
            GrowthStrategyAnalyzer(),
            QualityStrategyAnalyzer(),
        ],
    )

    comparison.plot_wealth_index()
    comparison.plot_annual_returns()
    comparison.plot_risk_return()
    comparison.plot_excess_returns()
    comparison.plot_turnover()
    comparison.plot_strategy_overlap_summary()
    plt.show(block=True)

if __name__ == '__main__':
    main()

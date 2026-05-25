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

    print("\n\n -----================================ Factor-Based Investment Strategy Analyzer ================================----- \n")
    command = 0
    while(command != 4):

        print("================================================ Options: ================================================")
        print("1. Plots\n" \
            "2. Holdings tables\n" \
            "3. All data\n" \
            "4. Quit")
        
        try:
            command = int(input(">"))
        except ValueError:
            print("Enter number from 1 to 4\n")
            continue

        if(command == 4):
            break

        if(command == 1):
            print("================================================ Plots: ================================================")
            print("1. Wealth index plot\n" \
                "2. Annual returns plot\n" \
                "3. Risk / return plot\n" \
                "4. Excess returns plot\n" \
                "5. Turnover plot\n" \
                "6. Strategy overlap summary plot")
        
            try:
                command = int(input(">"))
            except ValueError:
                print("Enter number from 1 to 6\n")
                continue

            match [command]:
                case[1]:
                    comparison.plot_wealth_index()
                case[2]:
                    comparison.plot_annual_returns()
                case[3]:
                    comparison.plot_risk_return()
                case[4]:
                    comparison.plot_excess_returns()
                case[5]:
                    comparison.plot_turnover()
                case[6]:
                    comparison.plot_strategy_overlap_summary()
   
            continue
                
        if(command == 2):
            year = 2
            print("================================================ Select year: ================================================")
            print("1. 2023\n" \
                "2. 2024\n" \
                "3. 2025")
            try:
                year = int(input(">"))
            except ValueError:
                print("Enter number from 1 to 4\n")
                continue

            match[year]:
                case[1]:
                    year = 2023
                case[2]:
                    year = 2024
                case[3]:
                    year = 2025

            print("================================================ Holdings tables: ================================================")
            print("1. Wealth index\n" \
                "2. Annual returns\n" \
                "3. Rist / return\n" \
                "4. Excess returns\n" \
                "5. Turnover\n" \
                "6. Strategy overlap summary")
            
            try:
                command = int(input(">"))
            except ValueError:
                print("Enter number from 1 to 5\n")
                continue

            match [command]:
                case[1]:
                    comparison.plot_strategy_holdings_table(PassiveStrategyAnalyzer.STRATEGY_NAME, year)
                case[2]:
                    comparison.plot_strategy_holdings_table(MomentumStrategyAnalyzer.STRATEGY_NAME, year)
                case[3]:
                    comparison.plot_strategy_holdings_table(ValueStrategyAnalyzer.STRATEGY_NAME, year)
                case[4]:
                    comparison.plot_strategy_holdings_table(GrowthStrategyAnalyzer.STRATEGY_NAME, year)
                case[5]:
                    comparison.plot_strategy_holdings_table(QualityStrategyAnalyzer.STRATEGY_NAME, year)
            
            continue

        if(command == 3):
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

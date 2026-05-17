import matplotlib.pyplot as plt
import pandas as pd

from analyzers import StrategyAnalyzerBase


class StrategyComparisonPlotter:
    """"""

    COMMON_WEALTH_INDEX_COLUMN: str = 'COMMON_WEALTH_INDEX'
    COMMON_CUMULATIVE_RETURN_COLUMN: str = 'COMMON_CUMULATIVE_RETURN'

    def __init__(self, analyzers: list[StrategyAnalyzerBase]) -> None:
        """"""

        self.analyzers = analyzers

    def build_comparison_result(self) -> pd.DataFrame:
        """"""

        frames = []

        for analyzer in self.analyzers:

            result = analyzer.analyze()

            if not result.empty:
                frames.append(result)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def get_common_years(self, result: pd.DataFrame) -> list[int]:
        """"""

        if result.empty:
            return []

        strategy_years = [set(strategy_result['YEAR']) for _, strategy_result in result.groupby('STRATEGY')]

        if not strategy_years:
            return []

        return sorted(set.intersection(*strategy_years))

    def build_common_period_result(self) -> pd.DataFrame:
        """"""

        result = self.build_comparison_result()

        if result.empty:
            return pd.DataFrame()

        common_years = self.get_common_years(result)

        if not common_years:
            return pd.DataFrame()

        common_result = result[result['YEAR'].isin(common_years)].copy()
        common_result = common_result.sort_values(['STRATEGY', 'YEAR'])

        common_result[self.COMMON_WEALTH_INDEX_COLUMN] = common_result.groupby('STRATEGY')['RETURN'].transform(
            lambda returns: (1 + returns).cumprod()
        )

        common_result[self.COMMON_CUMULATIVE_RETURN_COLUMN] = common_result[self.COMMON_WEALTH_INDEX_COLUMN] - 1

        return common_result

    def plot_wealth_index(self) -> None:
        """"""

        result = self.build_common_period_result()

        if result.empty:
            return

        common_years = sorted(result['YEAR'].unique())

        plt.figure(figsize=(10, 5))

        for strategy_name, strategy_result_raw in result.groupby('STRATEGY'):

            strategy_result = strategy_result_raw.sort_values('YEAR')

            plt.plot(
                strategy_result['YEAR'],
                strategy_result[self.COMMON_WEALTH_INDEX_COLUMN],
                marker='o',
                label=strategy_name,
            )

        plt.title('Strategy comparison: wealth index on common period')
        plt.xlabel('Year')
        plt.ylabel('Wealth index')

        plt.grid(visible=True)
        plt.legend()
        plt.xticks(common_years)

        plt.tight_layout()
        plt.show()

    def plot_annual_returns(self) -> None:
        """"""

        result = self.build_common_period_result()

        if result.empty:
            return

        years = sorted(result['YEAR'].unique())
        strategies = sorted(result['STRATEGY'].unique())

        bar_width = 0.8 / len(strategies)

        x_positions = list(range(len(years)))

        plt.figure(figsize=(10, 5))

        for strategy_index, strategy_name in enumerate(strategies):

            strategy_result = result[result['STRATEGY'] == strategy_name].set_index('YEAR').reindex(years)

            bar_positions = [
                x_position - 0.4 + bar_width / 2 + strategy_index * bar_width
                for x_position in x_positions
            ]

            plt.bar(
                bar_positions,
                strategy_result['RETURN'] * 100,
                width=bar_width,
                label=strategy_name,
            )

        plt.title('Strategy comparison: annual returns on common period')
        plt.xlabel('Year')
        plt.ylabel('Return, %')

        plt.axhline(0, linewidth=1)
        plt.grid(visible=True, axis='y')
        plt.legend()
        plt.xticks(x_positions, years)

        plt.tight_layout()
        plt.show()

    def build_risk_return_result(self) -> pd.DataFrame:
        """"""

        result = self.build_common_period_result()

        if result.empty:
            return pd.DataFrame()

        risk_return_result = result.groupby('STRATEGY').agg(
            AVG_ANNUAL_RETURN=('RETURN', 'mean'),
            VOLATILITY=('RETURN', 'std'),
            YEARS_COUNT=('YEAR', 'count'),
            FINAL_WEALTH_INDEX=(self.COMMON_WEALTH_INDEX_COLUMN, 'last'),
            CUMULATIVE_RETURN=(self.COMMON_CUMULATIVE_RETURN_COLUMN, 'last'),
        ).reset_index()

        risk_return_result['RETURN_TO_RISK'] = (
            risk_return_result['AVG_ANNUAL_RETURN'] / risk_return_result['VOLATILITY']
        )

        return risk_return_result.sort_values('RETURN_TO_RISK', ascending=False)

    def plot_risk_return(self) -> None:
        """"""

        result = self.build_risk_return_result()

        if result.empty:
            return

        plt.figure(figsize=(9, 6))

        plt.scatter(
            result['VOLATILITY'] * 100,
            result['AVG_ANNUAL_RETURN'] * 100,
            s=100,
        )

        for _, row in result.iterrows():
            plt.annotate(
                row['STRATEGY'],
                (
                    row['VOLATILITY'] * 100,
                    row['AVG_ANNUAL_RETURN'] * 100,
                ),
                textcoords='offset points',
                xytext=(8, 6),
            )

        plt.title('Strategy comparison: risk and return')
        plt.xlabel('Volatility of annual returns, %')
        plt.ylabel('Average annual return, %')

        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        plt.grid(visible=True)

        plt.tight_layout()
        plt.show()

    def build_excess_return_result(self) -> pd.DataFrame:
        """"""

        result = self.build_common_period_result()

        if result.empty:
            return pd.DataFrame()

        passive_result = result[result['STRATEGY'] == 'Passive'][['YEAR', 'RETURN']].rename(
            columns={'RETURN': 'PASSIVE_RETURN'}
        )

        if passive_result.empty:
            return pd.DataFrame()

        excess_result = result.merge(
            passive_result,
            on='YEAR',
            how='inner',
        )

        excess_result['EXCESS_RETURN'] = excess_result['RETURN'] - excess_result['PASSIVE_RETURN']

        return excess_result[excess_result['STRATEGY'] != 'Passive'].copy()

    def plot_excess_returns(self) -> None:
        """"""

        result = self.build_excess_return_result()

        if result.empty:
            return

        years = sorted(result['YEAR'].unique())
        strategies = sorted(result['STRATEGY'].unique())

        bar_width = 0.8 / len(strategies)

        x_positions = list(range(len(years)))

        plt.figure(figsize=(10, 5))

        for strategy_index, strategy_name in enumerate(strategies):

            strategy_result = result[result['STRATEGY'] == strategy_name].set_index('YEAR').reindex(years)

            bar_positions = [
                x_position - 0.4 + bar_width / 2 + strategy_index * bar_width
                for x_position in x_positions
            ]

            plt.bar(
                bar_positions,
                strategy_result['EXCESS_RETURN'] * 100,
                width=bar_width,
                label=strategy_name,
            )

        plt.title('Strategy comparison: excess return against Passive')
        plt.xlabel('Year')
        plt.ylabel('Excess return, p.p.')

        plt.axhline(0, linewidth=1)
        plt.grid(visible=True, axis='y')
        plt.legend()
        plt.xticks(x_positions, years)

        plt.tight_layout()
        plt.show()

    def build_holdings_result(self) -> pd.DataFrame:
        """"""

        frames = []

        for analyzer in self.analyzers:

            holdings = analyzer.analyze_holdings()

            if holdings.empty:
                continue

            holdings = holdings.copy()
            holdings['STRATEGY'] = analyzer.STRATEGY_NAME

            frames.append(holdings)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def build_turnover_result(self) -> pd.DataFrame:
        """"""

        holdings = self.build_holdings_result()

        if holdings.empty:
            return pd.DataFrame()

        result = self.build_common_period_result()

        if result.empty:
            return pd.DataFrame()

        common_years = sorted(result['YEAR'].unique())

        holdings = holdings[holdings['YEAR'].isin(common_years)].copy()

        rows = []

        for strategy_name, strategy_holdings in holdings.groupby('STRATEGY'):

            strategy_years = sorted(strategy_holdings['YEAR'].unique())

            previous_secids = None

            for year in strategy_years:

                current_secids = set(strategy_holdings[strategy_holdings['YEAR'] == year]['SECID'])

                if previous_secids is None:

                    previous_secids = current_secids
                    continue

                common_secids = current_secids & previous_secids
                new_secids = current_secids - previous_secids
                removed_secids = previous_secids - current_secids

                rows.append({
                    'STRATEGY': strategy_name,
                    'YEAR': year,
                    'PREVIOUS_COUNT': len(previous_secids),
                    'CURRENT_COUNT': len(current_secids),
                    'COMMON_COUNT': len(common_secids),
                    'NEW_COUNT': len(new_secids),
                    'REMOVED_COUNT': len(removed_secids),
                    'NEW_HOLDINGS_SHARE': (
                        len(new_secids) / len(current_secids)
                        if current_secids
                        else None
                    ),
                    'TURNOVER': (
                        1 - len(common_secids) / len(current_secids)
                        if current_secids
                        else None
                    ),
                })

                previous_secids = current_secids

        return pd.DataFrame(rows)

    def plot_turnover(self) -> None:
        """"""

        result = self.build_turnover_result()

        if result.empty:
            return

        years = sorted(result['YEAR'].unique())
        strategies = sorted(result['STRATEGY'].unique())

        bar_width = 0.8 / len(strategies)

        x_positions = list(range(len(years)))

        plt.figure(figsize=(10, 5))

        for strategy_index, strategy_name in enumerate(strategies):

            strategy_result = result[result['STRATEGY'] == strategy_name].set_index('YEAR').reindex(years)

            bar_positions = [
                x_position - 0.4 + bar_width / 2 + strategy_index * bar_width
                for x_position in x_positions
            ]

            plt.bar(
                bar_positions,
                strategy_result['TURNOVER'] * 100,
                width=bar_width,
                label=strategy_name,
            )

        plt.title('Strategy comparison: portfolio turnover')
        plt.xlabel('Year')
        plt.ylabel('Turnover, %')

        plt.grid(visible=True, axis='y')
        plt.legend()
        plt.xticks(x_positions, years)

        plt.tight_layout()
        plt.show()

    def build_strategy_overlap_result(self) -> pd.DataFrame:
        """"""

        holdings = self.build_holdings_result()

        if holdings.empty:
            return pd.DataFrame()

        common_result = self.build_common_period_result()

        if common_result.empty:
            return pd.DataFrame()

        common_years = sorted(common_result['YEAR'].unique())

        holdings = holdings[holdings['YEAR'].isin(common_years)].copy()

        if holdings.empty:
            return pd.DataFrame()

        rows = []

        for year, year_holdings in holdings.groupby('YEAR'):

            strategy_names = sorted(year_holdings['STRATEGY'].unique())

            for first_index, first_strategy_name in enumerate(strategy_names):

                first_secids = set(year_holdings[year_holdings['STRATEGY'] == first_strategy_name]['SECID'])

                for second_strategy_name in strategy_names[first_index + 1:]:

                    second_secids = set(year_holdings[year_holdings['STRATEGY'] == second_strategy_name]['SECID'])

                    common_secids = first_secids & second_secids
                    union_secids = first_secids | second_secids

                    rows.append(
                        {
                            'YEAR': year,
                            'STRATEGY_A': first_strategy_name,
                            'STRATEGY_B': second_strategy_name,
                            'COUNT_A': len(first_secids),
                            'COUNT_B': len(second_secids),
                            'COMMON_COUNT': len(common_secids),
                            'UNION_COUNT': len(union_secids),
                            'JACCARD_OVERLAP': len(common_secids) / len(union_secids) if union_secids else None,
                            'COMMON_SECIDS': ', '.join(sorted(common_secids)),
                        }
                    )

        return pd.DataFrame(rows)

    def build_strategy_overlap_summary(self) -> pd.DataFrame:
        """"""

        result = self.build_strategy_overlap_result()

        if result.empty:
            return pd.DataFrame()

        summary = result.groupby(['STRATEGY_A', 'STRATEGY_B']).agg(
            AVG_JACCARD_OVERLAP=('JACCARD_OVERLAP', 'mean'),
            MAX_JACCARD_OVERLAP=('JACCARD_OVERLAP', 'max'),
            MIN_JACCARD_OVERLAP=('JACCARD_OVERLAP', 'min'),
            AVG_COMMON_COUNT=('COMMON_COUNT', 'mean'),
            YEARS_COUNT=('YEAR', 'count'),
        ).reset_index()

        return summary.sort_values(
            [
                'AVG_JACCARD_OVERLAP',
                'AVG_COMMON_COUNT',
            ],
            ascending=False,
        )

    def plot_strategy_overlap_summary(self) -> None:
        """"""

        result = self.build_strategy_overlap_summary()

        if result.empty:
            return

        result = result.copy()

        result['PAIR'] = result['STRATEGY_A'] + ' / ' + result['STRATEGY_B']

        plt.figure(figsize=(10, 5))

        plt.bar(
            result['PAIR'],
            result['AVG_JACCARD_OVERLAP'] * 100,
        )

        plt.title('Average portfolio overlap between strategies')
        plt.xlabel('Strategy pair')
        plt.ylabel('Average Jaccard overlap, %')

        plt.grid(visible=True, axis='y')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

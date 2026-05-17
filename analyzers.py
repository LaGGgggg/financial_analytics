from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from settings import HISTORY_JSON_PATH, MSFO_DATA_JSON_PATH, TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH


@dataclass()
class RebalancePeriod:
    """"""

    year: int
    start_date: str
    end_date: str


class StrategyAnalyzerBase(ABC):
    """"""

    STRATEGY_NAME: str
    REBALANCE_PERIODS: tuple[RebalancePeriod, ...] = (
        RebalancePeriod(year=2021, start_date='2020-12-30', end_date='2021-12-30'),
        RebalancePeriod(year=2022, start_date='2021-12-30', end_date='2022-12-30'),
        RebalancePeriod(year=2023, start_date='2022-12-30', end_date='2023-12-29'),
        RebalancePeriod(year=2024, start_date='2023-12-29', end_date='2024-12-30'),
        RebalancePeriod(year=2025, start_date='2024-12-30', end_date='2025-12-30'),
    )
    TOP_FRACTION: float = 0.2

    @abstractmethod
    def build_period_returns(self) -> pd.DataFrame:
        """"""

    def load_securities(self) -> pd.DataFrame:
        """"""

        securities = pd.read_json(TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH)

        return securities[securities['LISTLEVEL'].isin((1, 2))].copy()

    def load_history(self) -> pd.DataFrame:
        """"""

        return pd.read_json(HISTORY_JSON_PATH)

    def load_prices(self) -> pd.DataFrame:
        """"""

        securities = self.load_securities()
        history = self.load_history()

        prices = history.merge(
            securities[['SECID', 'SECNAME', 'LISTLEVEL']],
            on='SECID',
            how='inner',
        )

        return prices[
            prices['SECID'].notna()
            & prices['TRADEDATE'].notna()
            & prices['CLOSE'].notna()
            & (prices['CLOSE'] > 0)
        ].copy()

    def load_fundamentals(self) -> pd.DataFrame:
        """"""

        fundamentals = pd.read_json(MSFO_DATA_JSON_PATH)

        fundamentals['YEAR'] = pd.to_numeric(fundamentals['YEAR'], errors='coerce')
        fundamentals['VALUE'] = pd.to_numeric(fundamentals['VALUE'], errors='coerce')

        return fundamentals[
            fundamentals['SECID'].notna()
            & fundamentals['YEAR'].notna()
            & fundamentals['METRIC'].notna()
        ].copy()

    def build_fundamentals_wide(self) -> pd.DataFrame:
        """"""

        fundamentals = self.load_fundamentals()

        fundamentals_wide = fundamentals.pivot_table(
            index=['SECID', 'YEAR'],
            columns='METRIC',
            values='VALUE',
            aggfunc='first',
        ).reset_index()

        fundamentals_wide.columns.name = None
        fundamentals_wide['YEAR'] = fundamentals_wide['YEAR'].astype(int)

        return fundamentals_wide

    def build_all_period_returns(self) -> pd.DataFrame:
        """"""

        prices = self.load_prices()
        frames = []

        for period in self.REBALANCE_PERIODS:

            start_prices = prices[prices['TRADEDATE'] == period.start_date].copy()
            end_prices = prices[prices['TRADEDATE'] == period.end_date].copy()

            start_prices = start_prices.rename(columns={
                'TRADEDATE': 'START_DATE',
                'CLOSE': 'START_CLOSE',
            })

            end_prices = end_prices.rename(columns={
                'TRADEDATE': 'END_DATE',
                'CLOSE': 'END_CLOSE',
            })

            period_returns = start_prices[[
                'SECID',
                'SHORTNAME',
                'SECNAME',
                'LISTLEVEL',
                'START_DATE',
                'START_CLOSE',
            ]].merge(
                end_prices[[
                    'SECID',
                    'END_DATE',
                    'END_CLOSE',
                ]],
                on='SECID',
                how='inner',
            )

            period_returns['YEAR'] = period.year
            period_returns['RETURN'] = period_returns['END_CLOSE'] / period_returns['START_CLOSE'] - 1

            period_returns = period_returns[
                period_returns['RETURN'].notna()
                & period_returns['START_CLOSE'].gt(0)
                & period_returns['END_CLOSE'].gt(0)
                & period_returns['RETURN'].between(-0.95, 10)
            ].copy()

            frames.append(period_returns)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def select_top_by_score(
        self,
        data: pd.DataFrame,
        score_column: str,
    ) -> pd.DataFrame:
        """"""

        data = data[data[score_column].notna()].copy()

        if data.empty:
            return pd.DataFrame()

        selected_count = max(1, int(len(data) * self.TOP_FRACTION))

        return data.sort_values(score_column, ascending=False).head(selected_count).copy()

    def analyze(self) -> pd.DataFrame:
        """"""

        returns = self.build_period_returns()

        if returns.empty:
            return pd.DataFrame()

        result = returns.groupby('YEAR', as_index=False).agg(
            STRATEGY=('YEAR', lambda _: self.STRATEGY_NAME),
            RETURN=('RETURN', 'mean'),
            COUNT=('SECID', 'count'),
        )

        result['WEALTH_INDEX'] = (1 + result['RETURN']).cumprod()
        result['CUMULATIVE_RETURN'] = result['WEALTH_INDEX'] - 1

        return result[[
            'YEAR',
            'STRATEGY',
            'RETURN',
            'COUNT',
            'WEALTH_INDEX',
            'CUMULATIVE_RETURN',
        ]]

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        return period_returns[[
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]].sort_values(['YEAR', 'SECID'])

    def print_result(self) -> None:
        """"""

        result = self.analyze()

        if result.empty:
            print('No data for analysis')
            return

        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 200,
        ):
            print(result.to_string(index=False))


class PassiveStrategyAnalyzer(StrategyAnalyzerBase):
    """"""

    STRATEGY_NAME: str = 'Passive'

    def build_period_returns(self) -> pd.DataFrame:
        """"""

        prices = self.load_prices()
        frames = []

        for period in self.REBALANCE_PERIODS:

            start_prices = prices[prices['TRADEDATE'] == period.start_date].copy()
            end_prices = prices[prices['TRADEDATE'] == period.end_date].copy()

            start_prices = start_prices.rename(columns={
                'TRADEDATE': 'START_DATE',
                'CLOSE': 'START_CLOSE',
            })

            end_prices = end_prices.rename(columns={
                'TRADEDATE': 'END_DATE',
                'CLOSE': 'END_CLOSE',
            })

            period_returns = start_prices[[
                'SECID',
                'SHORTNAME',
                'SECNAME',
                'LISTLEVEL',
                'START_DATE',
                'START_CLOSE',
            ]].merge(
                end_prices[[
                    'SECID',
                    'END_DATE',
                    'END_CLOSE',
                ]],
                on='SECID',
                how='inner',
            )

            period_returns['YEAR'] = period.year
            period_returns['RETURN'] = period_returns['END_CLOSE'] / period_returns['START_CLOSE'] - 1

            period_returns = period_returns[
                period_returns['RETURN'].notna()
                & period_returns['START_CLOSE'].gt(0)
                & period_returns['END_CLOSE'].gt(0)
                & period_returns['RETURN'].between(-0.95, 10)
            ].copy()

            frames.append(period_returns)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze(self) -> pd.DataFrame:
        """"""

        returns = self.build_period_returns()

        if returns.empty:
            return pd.DataFrame()

        result = returns.groupby('YEAR', as_index=False).agg(
            STRATEGY=('YEAR', lambda _: 'Passive'),
            RETURN=('RETURN', 'mean'),
            COUNT=('SECID', 'count'),
        )

        result['WEALTH_INDEX'] = (1 + result['RETURN']).cumprod()
        result['CUMULATIVE_RETURN'] = result['WEALTH_INDEX'] - 1

        return result[[
            'YEAR',
            'STRATEGY',
            'RETURN',
            'COUNT',
            'WEALTH_INDEX',
            'CUMULATIVE_RETURN',
        ]]

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        return period_returns[[
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]].sort_values(['YEAR', 'SECID'])

    def print_result(self) -> None:
        """"""

        result = self.analyze()

        if result.empty:
            print('No data for analysis')
            return

        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 200,
        ):
            print(result.to_string(index=False))


class MomentumStrategyAnalyzer(StrategyAnalyzerBase):
    """"""

    STRATEGY_NAME: str = 'Momentum'

    def build_period_returns(self) -> pd.DataFrame:
        """"""

        all_returns = self.build_all_period_returns()

        if all_returns.empty:
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:

            previous_year = period.year - 1

            previous_returns = all_returns[all_returns['YEAR'] == previous_year].copy()
            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()

            if previous_returns.empty or current_returns.empty:
                continue

            previous_returns = previous_returns.rename(columns={'RETURN': 'MOMENTUM'})

            momentum_table = previous_returns[[
                'SECID',
                'MOMENTUM',
            ]].merge(
                current_returns,
                on='SECID',
                how='inner',
            )

            momentum_table = momentum_table[momentum_table['MOMENTUM'].notna()].copy()

            if momentum_table.empty:
                continue

            selected_count = max(1, int(len(momentum_table) * self.TOP_FRACTION))

            momentum_table = momentum_table.sort_values('MOMENTUM', ascending=False).head(selected_count).copy()

            frames.append(momentum_table)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        return period_returns[[
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'MOMENTUM',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]].sort_values(['YEAR', 'MOMENTUM'], ascending=[True, False])


class ValueStrategyAnalyzer(StrategyAnalyzerBase):
    """"""

    STRATEGY_NAME: str = 'Value'
    VALUE_METRICS: tuple[str, ...] = (
        'P/E',
        'P/BV',
        'EV/EBITDA',
    )

    def add_value_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """"""

        data = data.copy()

        score_columns = []

        for metric in self.VALUE_METRICS:

            if metric not in data.columns:
                continue

            score_column = f'{metric}_VALUE_SCORE'

            valid_values = data[metric].notna() & data[metric].gt(0)

            data[score_column] = None

            data.loc[valid_values, score_column] = (1 / data.loc[valid_values, metric]).rank(pct=True)

            score_columns.append(score_column)

        if not score_columns:
            data['VALUE_SCORE'] = None
            return data

        data['VALUE_SCORE'] = data[score_columns].mean(axis=1, skipna=True)

        return data

    def build_period_returns(self) -> pd.DataFrame:
        """"""

        all_returns = self.build_all_period_returns()
        fundamentals_wide = self.build_fundamentals_wide()

        if all_returns.empty or fundamentals_wide.empty:
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:

            signal_year = period.year - 1

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            signals = fundamentals_wide[fundamentals_wide['YEAR'] == signal_year].copy()

            if current_returns.empty or signals.empty:
                continue

            signals = signals.rename(columns={
                'YEAR': 'SIGNAL_YEAR',
            })

            value_table = current_returns.merge(signals, on='SECID', how='inner')

            if value_table.empty:
                continue

            value_table = self.add_value_score(value_table)
            value_table = self.select_top_by_score(data=value_table, score_column='VALUE_SCORE')

            if value_table.empty:
                continue

            frames.append(value_table)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'SIGNAL_YEAR',
            'VALUE_SCORE',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        for metric in self.VALUE_METRICS:
            if metric in period_returns.columns:
                columns.append(metric)  # noqa: PERF401

        return period_returns[columns].sort_values(
            ['YEAR', 'VALUE_SCORE'],
            ascending=[True, False],
        )


class GrowthStrategyAnalyzer(StrategyAnalyzerBase):
    """"""

    STRATEGY_NAME: str = 'Growth'
    GROWTH_METRICS: tuple[str, ...] = (
        'Выручка ,  млрд руб', 'EBITDA ,  млрд руб', 'Чистая прибыль ,  млрд руб'    # noqa: RUF001
    )

    def add_growth_metrics(self, fundamentals_wide: pd.DataFrame) -> pd.DataFrame:
        """"""

        fundamentals_wide = fundamentals_wide.copy()
        fundamentals_wide = fundamentals_wide.sort_values(['SECID', 'YEAR'])

        growth_columns = []

        for metric in self.GROWTH_METRICS:

            if metric not in fundamentals_wide.columns:
                continue

            growth_column = f'{metric}_GROWTH'

            previous_value = fundamentals_wide.groupby('SECID')[metric].shift(1)

            fundamentals_wide[growth_column] = fundamentals_wide[metric] / previous_value - 1

            fundamentals_wide.loc[
                previous_value.isna()
                | previous_value.eq(0)
                | fundamentals_wide[metric].isna(),
                growth_column,
            ] = None

            growth_columns.append(growth_column)

        if not growth_columns:
            fundamentals_wide['GROWTH_SCORE'] = None
            return fundamentals_wide

        score_columns = []

        for growth_column in growth_columns:

            score_column = f'{growth_column}_SCORE'

            fundamentals_wide[score_column] = fundamentals_wide[growth_column].rank(pct=True)

            score_columns.append(score_column)

        fundamentals_wide['GROWTH_SCORE'] = fundamentals_wide[score_columns].mean(axis=1, skipna=True)

        return fundamentals_wide

    def build_period_returns(self) -> pd.DataFrame:
        """"""

        all_returns = self.build_all_period_returns()
        fundamentals_wide = self.build_fundamentals_wide()

        if all_returns.empty or fundamentals_wide.empty:
            return pd.DataFrame()

        fundamentals_wide = self.add_growth_metrics(fundamentals_wide)

        frames = []

        for period in self.REBALANCE_PERIODS:

            signal_year = period.year - 1

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            signals = fundamentals_wide[fundamentals_wide['YEAR'] == signal_year].copy()

            if current_returns.empty or signals.empty:
                continue

            signals = signals.rename(columns={'YEAR': 'SIGNAL_YEAR'})

            growth_table = current_returns.merge(signals, on='SECID', how='inner')

            if growth_table.empty:
                continue

            growth_table = self.select_top_by_score(data=growth_table, score_column='GROWTH_SCORE')

            if growth_table.empty:
                continue

            frames.append(growth_table)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'SIGNAL_YEAR',
            'GROWTH_SCORE',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        for metric in self.GROWTH_METRICS:

            if metric in period_returns.columns:
                columns.append(metric)

            growth_column = f'{metric}_GROWTH'

            if growth_column in period_returns.columns:
                columns.append(growth_column)

        return period_returns[columns].sort_values(
            ['YEAR', 'GROWTH_SCORE'],
            ascending=[True, False],
        )


class QualityStrategyAnalyzer(StrategyAnalyzerBase):
    """"""

    STRATEGY_NAME: str = 'Quality'
    NET_DEBT_METRIC: str = 'Чистый долг ,  млрд руб'  # noqa: RUF001
    EBITDA_METRIC: str = 'EBITDA'
    NET_DEBT_TO_EBITDA_METRIC: str = 'Чистый долг/EBITDA'
    QUALITY_HIGHER_IS_BETTER_METRICS: tuple[str, ...] = (
        'ROE ,  %',
        'ROA ,  %',
        'Рентаб EBITDA ,  %',
        'Чистая рентаб ,  %',
    )
    QUALITY_LOWER_IS_BETTER_METRICS: tuple[str, ...] = (
        NET_DEBT_TO_EBITDA_METRIC,
    )

    def add_calculated_quality_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """"""

        data = data.copy()

        if self.NET_DEBT_METRIC in data.columns and self.EBITDA_METRIC in data.columns:

            valid_values = (
                data[self.NET_DEBT_METRIC].notna()
                & data[self.EBITDA_METRIC].notna()
                & data[self.EBITDA_METRIC].ne(0)
            )

            data[self.NET_DEBT_TO_EBITDA_METRIC] = None

            data.loc[valid_values, self.NET_DEBT_TO_EBITDA_METRIC] = (
                data.loc[valid_values, self.NET_DEBT_METRIC] / data.loc[valid_values, self.EBITDA_METRIC]
            )

        return data

    def add_quality_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """"""

        data = data.copy()
        data = self.add_calculated_quality_metrics(data)

        score_columns = []

        for metric in self.QUALITY_HIGHER_IS_BETTER_METRICS:

            if metric not in data.columns:
                continue

            score_column = f'{metric}_QUALITY_SCORE'

            data[score_column] = data[metric].rank(pct=True)

            score_columns.append(score_column)

        for metric in self.QUALITY_LOWER_IS_BETTER_METRICS:

            if metric not in data.columns:
                continue

            score_column = f'{metric}_QUALITY_SCORE'

            data[score_column] = (-data[metric]).rank(pct=True)

            score_columns.append(score_column)

        if not score_columns:

            data['QUALITY_SCORE'] = None
            return data

        data['QUALITY_SCORE'] = data[score_columns].mean(axis=1, skipna=True)

        return data

    def build_period_returns(self) -> pd.DataFrame:
        """"""

        all_returns = self.build_all_period_returns()
        fundamentals_wide = self.build_fundamentals_wide()

        if all_returns.empty or fundamentals_wide.empty:
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:

            signal_year = period.year - 1

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            signals = fundamentals_wide[fundamentals_wide['YEAR'] == signal_year].copy()

            if current_returns.empty or signals.empty:
                continue

            signals = signals.rename(columns={'YEAR': 'SIGNAL_YEAR'})

            quality_table = current_returns.merge(signals, on='SECID', how='inner')

            if quality_table.empty:
                continue

            quality_table = self.add_quality_score(quality_table)

            quality_table = self.select_top_by_score(data=quality_table, score_column='QUALITY_SCORE')

            if quality_table.empty:
                continue

            frames.append(quality_table)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """"""

        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'SIGNAL_YEAR',
            'QUALITY_SCORE',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        for metric in self.QUALITY_HIGHER_IS_BETTER_METRICS:
            if metric in period_returns.columns:
                columns.append(metric)  # noqa: PERF401

        for metric in (
            self.NET_DEBT_METRIC,
            self.EBITDA_METRIC,
            self.NET_DEBT_TO_EBITDA_METRIC,
        ):
            if metric in period_returns.columns:
                columns.append(metric)  # noqa: PERF401

        return period_returns[columns].sort_values(
            ['YEAR', 'QUALITY_SCORE'],
            ascending=[True, False],
        )

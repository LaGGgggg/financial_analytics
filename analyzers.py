from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import numpy as np

from settings import HISTORY_JSON_PATH, MSFO_DATA_JSON_PATH, TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH, DAILY_HISTORY_JSON_PATH


@dataclass()
class RebalancePeriod:

    year: int
    start_date: str
    end_date: str


class StrategyAnalyzerBase(ABC):

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
        pass

    def load_securities(self) -> pd.DataFrame:

        securities = pd.read_json(TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH)

        return securities[securities['LISTLEVEL'].isin((1, 2))].copy()

    def load_history(self) -> pd.DataFrame:

        return pd.read_json(HISTORY_JSON_PATH)

    def load_prices(self) -> pd.DataFrame:

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

        fundamentals = pd.read_json(MSFO_DATA_JSON_PATH)

        fundamentals['YEAR'] = pd.to_numeric(fundamentals['YEAR'], errors='coerce')
        fundamentals['VALUE'] = pd.to_numeric(fundamentals['VALUE'], errors='coerce')

        return fundamentals[
            fundamentals['SECID'].notna()
            & fundamentals['YEAR'].notna()
            & fundamentals['METRIC'].notna()
        ].copy()

    def build_fundamentals_wide(self) -> pd.DataFrame:

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

        data = data[data[score_column].notna()].copy()

        if data.empty:
            return pd.DataFrame()

        selected_count = max(1, int(len(data) * self.TOP_FRACTION))

        return data.sort_values(score_column, ascending=False).head(selected_count).copy()

    def analyze(self) -> pd.DataFrame:

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

        result = self.analyze()

        if result.empty:
            print('No data for analysis')  # noqa: T201
            return

        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 200,
        ):
            print(result.to_string(index=False))  # noqa: T201


class PassiveStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Passive'

    def build_period_returns(self) -> pd.DataFrame:

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

        result = self.analyze()

        if result.empty:
            print('No data for analysis')  # noqa: T201
            return

        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 200,
        ):
            print(result.to_string(index=False))  # noqa: T201


class MomentumStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Momentum'

    def build_period_returns(self) -> pd.DataFrame:

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

    STRATEGY_NAME: str = 'Value'
    VALUE_METRICS: tuple[str, ...] = (
        'P/E',
        'P/BV',
        'EV/EBITDA',
    )

    def add_value_score(self, data: pd.DataFrame) -> pd.DataFrame:

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

    STRATEGY_NAME: str = 'Growth'
    GROWTH_METRICS: tuple[str, ...] = (
        'Выручка ,  млрд руб', 'EBITDA ,  млрд руб', 'Чистая прибыль ,  млрд руб'    # noqa: RUF001
    )

    def add_growth_metrics(self, fundamentals_wide: pd.DataFrame) -> pd.DataFrame:

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


class SeasonalityStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Seasonality'
    MIN_YEARS_HISTORY: int = 3

    def build_period_returns(self) -> pd.DataFrame:
        """
        Для каждого периода ребалансировки:
        1. Определяем месяц входа (start_date) и месяц выхода (end_date)
        2. Рассчитываем историческую доходность для этой пары месяцев
        3. Выбираем топ-20% акций с наилучшей исторической сезонной доходностью
        """
        all_prices = self.load_prices()
        all_returns = self.build_all_period_returns()

        if all_prices.empty or all_returns.empty:
            return pd.DataFrame()

        all_prices['TRADEDATE'] = pd.to_datetime(all_prices['TRADEDATE'])
        all_prices['MONTH'] = all_prices['TRADEDATE'].dt.month
        all_prices['YEAR'] = all_prices['TRADEDATE'].dt.year

        frames = []

        for period in self.REBALANCE_PERIODS:
            entry_month = pd.to_datetime(period.start_date).month
            exit_month = pd.to_datetime(period.end_date).month

            # Рассчитываем историческую сезонную доходность для каждого тикера
            seasonal_scores = self._calculate_seasonal_scores(
                all_prices, entry_month, exit_month, period.year
            )

            if seasonal_scores.empty:
                continue

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()

            if current_returns.empty:
                continue

            period_data = current_returns.merge(
                seasonal_scores,
                on='SECID',
                how='inner',
            )

            if period_data.empty:
                continue

            # Выбираем топ по сезонному скору
            selected = self.select_top_by_score(
                data=period_data,
                score_column='SEASONALITY_SCORE',
            )

            if selected.empty:
                continue

            frames.append(selected)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _calculate_seasonal_scores(
        self,
        prices: pd.DataFrame,
        entry_month: int,
        exit_month: int,
        current_year: int,
    ) -> pd.DataFrame:
        """
        Рассчитывает среднюю доходность для пары (entry_month -> exit_month)
        за все доступные исторические годы (кроме текущего).
        """
        historical_prices = prices[prices['YEAR'] < current_year].copy()

        if historical_prices.empty:
            return pd.DataFrame()

        monthly_prices = (
            historical_prices.groupby(['SECID', 'YEAR', 'MONTH'])['CLOSE']
            .mean()
            .reset_index()
        )

        seasonal_returns = []

        for secid in monthly_prices['SECID'].unique():
            sec_data = monthly_prices[monthly_prices['SECID'] == secid]

            for year in sec_data['YEAR'].unique():
                entry_price = sec_data[
                    (sec_data['YEAR'] == year) & (sec_data['MONTH'] == entry_month)
                ]['CLOSE'].values

                exit_price = sec_data[
                    (sec_data['YEAR'] == year) & (sec_data['MONTH'] == exit_month)
                ]['CLOSE'].values

                if len(entry_price) > 0 and len(exit_price) > 0:
                    entry_price = entry_price[0]
                    exit_price = exit_price[0]

                    if entry_price > 0:
                        ret = exit_price / entry_price - 1
                        seasonal_returns.append({
                            'SECID': secid,
                            'YEAR': year,
                            'SEASONAL_RETURN': ret,
                        })

        if not seasonal_returns:
            return pd.DataFrame()

        seasonal_df = pd.DataFrame(seasonal_returns)

        # Проверяем, что достаточно истории
        years_count = seasonal_df.groupby('SECID')['YEAR'].nunique()
        valid_secids = years_count[years_count >= self.MIN_YEARS_HISTORY].index

        if len(valid_secids) == 0:
            return pd.DataFrame()

        # Рассчитываем среднюю сезонную доходность для каждого тикера
        seasonal_scores = (
            seasonal_df[seasonal_df['SECID'].isin(valid_secids)]
            .groupby('SECID')['SEASONAL_RETURN']
            .mean()
            .reset_index()
            .rename(columns={'SEASONAL_RETURN': 'SEASONALITY_SCORE'})
        )

        return seasonal_scores

    def analyze_holdings(self) -> pd.DataFrame:
        """
        Финальная таблица с результатами стратегии.
        """
        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'SEASONALITY_SCORE',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        available_columns = [col for col in columns if col in period_returns.columns]

        return period_returns[available_columns].sort_values(
            ['YEAR', 'SEASONALITY_SCORE'],
            ascending=[True, False],
        )
class FractalStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Fractal'
    MIN_LAG: int = 10
    MAX_LAG: int = 100
    MIN_DATA_POINTS: int = 120

    def load_daily_history(self) -> pd.DataFrame:
        """Загружает предварительно рассчитанные ежедневные данные."""
        if not DAILY_HISTORY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Файл {DAILY_HISTORY_JSON_PATH} не найден. "
                "Запустите IssParser().parse_and_save() для его создания."
            )
        df = pd.read_json(DAILY_HISTORY_JSON_PATH)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        return df

    def _calculate_hurst(self, series: np.ndarray) -> float | None:
        """R/S-анализ. Возвращает показатель Хёрста H."""
        n = len(series)
        if n < self.MIN_DATA_POINTS:
            return None

        max_lag = min(self.MAX_LAG, n // 2)
        if max_lag <= self.MIN_LAG:
            return None

        lags = np.arange(self.MIN_LAG, max_lag + 1)
        rs_values = np.empty(len(lags))

        for i, lag in enumerate(lags):
            n_windows = n // lag
            if n_windows < 1:
                rs_values[i] = np.nan
                continue

            windows = series[: n_windows * lag].reshape(n_windows, lag)
            means = windows.mean(axis=1, keepdims=True)
            cumdev = np.cumsum(windows - means, axis=1)

            R = cumdev.max(axis=1) - cumdev.min(axis=1)
            S = windows.std(axis=1, ddof=1)

            valid = S > 1e-12
            if valid.any():
                rs_values[i] = np.mean(R[valid] / S[valid])
            else:
                rs_values[i] = np.nan

        valid_mask = (rs_values > 0) & np.isfinite(rs_values)
        if valid_mask.sum() < 3:
            return None

        try:
            slope, _ = np.polyfit(
                np.log(lags[valid_mask]),
                np.log(rs_values[valid_mask]),
                1,
            )
        except (np.linalg.LinAlgError, ValueError):
            return None

        if not np.isfinite(slope) or slope < 0 or slope > 1:
            return None

        return float(slope)

    def _calculate_fractal_scores(self, signal_date: str) -> pd.DataFrame:
        """Рассчитывает показатель Хёрста для каждого тикера."""
        daily_prices = self.load_daily_history()
        signal_dt = pd.to_datetime(signal_date)

        historical = daily_prices[daily_prices['TRADEDATE'] <= signal_dt].copy()
        
        if historical.empty:
            print(f'[Fractal] Набор данных пуст после фильтрации. Возможно, данные в файле новее, чем {signal_date}')
            return pd.DataFrame()

        historical = historical.sort_values(['SECID', 'TRADEDATE'])

        scores = []
        for secid, group in historical.groupby('SECID'):
            closes = group['CLOSE'].values.astype(float)
            # Логарифмические доходности
            log_returns = np.diff(np.log(closes))
            log_returns = log_returns[np.isfinite(log_returns)]

            if len(log_returns) < self.MIN_DATA_POINTS:
                continue

            h = self._calculate_hurst(log_returns)
            if h is not None:
                scores.append({
                    'SECID': secid,
                    'HURST_SCORE': h,
                    'DATA_POINTS': len(log_returns),
                })

        return pd.DataFrame(scores) if scores else pd.DataFrame()

    def build_period_returns(self) -> pd.DataFrame:
        all_returns = self.build_all_period_returns()
        if all_returns.empty:
            return pd.DataFrame()

        frames = []
        for period in self.REBALANCE_PERIODS:
            fractal_scores = self._calculate_fractal_scores(signal_date=period.start_date)
            if fractal_scores.empty:
                continue

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            if current_returns.empty:
                continue

            period_data = current_returns.merge(fractal_scores, on='SECID', how='inner')
            if period_data.empty:
                continue

            selected = self.select_top_by_score(data=period_data, score_column='HURST_SCORE')
            if not selected.empty:
                frames.append(selected)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def analyze_holdings(self) -> pd.DataFrame:
        period_returns = self.build_period_returns()
        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'HURST_SCORE',
            'DATA_POINTS',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]
        available = [c for c in columns if c in period_returns.columns]

        return period_returns[available].sort_values(['YEAR', 'HURST_SCORE'], ascending=[True, False])



class NetworkStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Network'
    CORRELATION_WINDOW: int = 252
    MIN_DATA_POINTS: int = 120
    USE_EIGENVECTOR: bool = False  # если True — eigenvector centrality, иначе — sum of |corr|

    def load_daily_history(self) -> pd.DataFrame:
        """Загружает ежедневные данные из кэша."""
        if not DAILY_HISTORY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Файл {DAILY_HISTORY_JSON_PATH} не найден. "
                "Запустите IssParser().parse_and_save() для его создания."
            )
        df = pd.read_json(DAILY_HISTORY_JSON_PATH)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        return df

    def _calculate_centrality_scores(
        self,
        daily_prices: pd.DataFrame,
        signal_date: str,
    ) -> pd.DataFrame:
        """
        Рассчитывает centrality score для каждого тикера на основе
        матрицы корреляций доходностей за окно CORRELATION_WINDOW.
        """
        signal_dt = pd.to_datetime(signal_date)

        historical = (
            daily_prices[daily_prices['TRADEDATE'] <= signal_dt]
            .copy()
            .sort_values(['SECID', 'TRADEDATE'])
        )

        if historical.empty:
            return pd.DataFrame()

        historical['rank'] = historical.groupby('SECID')['TRADEDATE'].rank(
            method='dense', ascending=False
        )
        historical = historical[historical['rank'] <= self.CORRELATION_WINDOW].copy()
        historical = historical.drop(columns=['rank'])

        if historical.empty:
            return pd.DataFrame()

        historical = historical.sort_values(['SECID', 'TRADEDATE'])
        historical['LOG_RETURN'] = historical.groupby('SECID')['CLOSE'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        returns_wide = historical.pivot_table(
            index='TRADEDATE',
            columns='SECID',
            values='LOG_RETURN',
        )

        valid_tickers = returns_wide.columns[
            returns_wide.count() >= self.MIN_DATA_POINTS
        ]

        if len(valid_tickers) < 2:
            return pd.DataFrame()

        returns_wide = returns_wide[valid_tickers]

        corr_matrix = returns_wide.corr()

        n = len(corr_matrix)
        mask = ~np.eye(n, dtype=bool)
        corr_matrix = corr_matrix.where(mask)

        if self.USE_EIGENVECTOR:
            # Для eigenvector centrality нужна numpy-матрица
            abs_corr = corr_matrix.abs().fillna(0).to_numpy()
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(abs_corr)
                principal = eigenvectors[:, -1]
                if principal.sum() < 0:
                    principal = -principal
                centrality = pd.Series(principal, index=valid_tickers)
            except np.linalg.LinAlgError:
                return pd.DataFrame()
        else:
            centrality = corr_matrix.abs().sum(axis=1)

        c_min, c_max = centrality.min(), centrality.max()
        if c_max > c_min:
            centrality = (centrality - c_min) / (c_max - c_min)

        result = centrality.reset_index()
        result.columns = ['SECID', 'CENTRALITY_SCORE']

        points_per_ticker = returns_wide.count().reset_index()
        points_per_ticker.columns = ['SECID', 'DATA_POINTS']
        result = result.merge(points_per_ticker, on='SECID', how='left')

        return result

    
    def build_period_returns(self) -> pd.DataFrame:
        """
        Для каждого периода ребалансировки:
        1. Строим матрицу корреляций на исторических данных до сигнала
        2. Рассчитываем centrality score для каждого тикера
        3. Выбираем топ-20% с наибольшей центральностью
        """
        daily_prices = self.load_daily_history()
        all_returns = self.build_all_period_returns()

        if daily_prices.empty or all_returns.empty:
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:
            centrality_scores = self._calculate_centrality_scores(
                daily_prices,
                signal_date=period.start_date,
            )

            if centrality_scores.empty:
                continue

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            if current_returns.empty:
                continue

            period_data = current_returns.merge(
                centrality_scores,
                on='SECID',
                how='inner',
            )

            if period_data.empty:
                continue

            selected = self.select_top_by_score(
                data=period_data,
                score_column='CENTRALITY_SCORE',
            )

            if selected.empty:
                continue

            frames.append(selected)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """Финальная таблица с результатами стратегии."""
        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'CENTRALITY_SCORE',
            'DATA_POINTS',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        available = [c for c in columns if c in period_returns.columns]

        return period_returns[available].sort_values(
            ['YEAR', 'CENTRALITY_SCORE'],
            ascending=[True, False],
        )


class LowVolatilityStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'LowVolatility'
    VOLATILITY_WINDOW: int = 252
    MIN_DATA_POINTS: int = 180

    def load_daily_history(self) -> pd.DataFrame:
        """Загружает ежедневные данные из кэша."""
        if not DAILY_HISTORY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Файл {DAILY_HISTORY_JSON_PATH} не найден. "
                "Запустите IssParser().parse_and_save() для его создания."
            )
        df = pd.read_json(DAILY_HISTORY_JSON_PATH)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        return df

    def _calculate_volatility_scores(
        self,
        daily_prices: pd.DataFrame,
        signal_date: str,
    ) -> pd.DataFrame:
        """
        Рассчитывает годовую волатильность для каждого тикера.

        Волатильность = std(дневных логарифмических доходностей) * sqrt(252)

        Чем НИЖЕ волатильность, тем ВЫШЕ скор (инвертируем для select_top_by_score).
        """
        signal_dt = pd.to_datetime(signal_date)

        historical = (
            daily_prices[daily_prices['TRADEDATE'] <= signal_dt]
            .copy()
            .sort_values(['SECID', 'TRADEDATE'])
        )

        if historical.empty:
            return pd.DataFrame()

        # Оставляем только последние VOLATILITY_WINDOW точек на тикер
        historical['rank'] = historical.groupby('SECID')['TRADEDATE'].rank(
            method='dense', ascending=False
        )
        historical = historical[historical['rank'] <= self.VOLATILITY_WINDOW].copy()
        historical = historical.drop(columns=['rank'])

        # Фильтруем тикеры с недостаточным количеством данных
        ticker_counts = historical.groupby('SECID')['TRADEDATE'].count()
        valid_tickers = ticker_counts[ticker_counts >= self.MIN_DATA_POINTS].index
        historical = historical[historical['SECID'].isin(valid_tickers)].copy()

        if historical.empty:
            return pd.DataFrame()

        # Считаем дневные логарифмические доходности
        historical['LOG_RETURN'] = historical.groupby('SECID')['CLOSE'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        # Годовая волатильность для каждого тикера
        volatility = (
            historical.groupby('SECID')['LOG_RETURN']
            .std()
            .multiply(np.sqrt(252))
            .reset_index()
        )
        volatility.columns = ['SECID', 'VOLATILITY']

        # Инвертируем: чем ниже волатильность, тем выше скор
        # (для совместимости с select_top_by_score, который выбирает по убыванию)
        vol_min = volatility['VOLATILITY'].min()
        vol_max = volatility['VOLATILITY'].max()

        if vol_max > vol_min:
            volatility['LOW_VOL_SCORE'] = (vol_max - volatility['VOLATILITY']) / (vol_max - vol_min)
        else:
            volatility['LOW_VOL_SCORE'] = 0.5

        # Добавляем количество точек для отладки
        volatility['DATA_POINTS'] = (
            historical.groupby('SECID')['LOG_RETURN'].count().reindex(volatility['SECID']).values
        )

        return volatility

    def build_period_returns(self) -> pd.DataFrame:
        """
        Для каждого периода ребалансировки:
        1. Рассчитываем волатильность за последние VOLATILITY_WINDOW дней
        2. Выбираем топ-20% акций с наименьшей волатильностью
        3. Берём их фактическую доходность за период
        """
        daily_prices = self.load_daily_history()
        all_returns = self.build_all_period_returns()

        if daily_prices.empty or all_returns.empty:
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:
            vol_scores = self._calculate_volatility_scores(
                daily_prices,
                signal_date=period.start_date,
            )

            if vol_scores.empty:
                continue

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            if current_returns.empty:
                continue

            period_data = current_returns.merge(
                vol_scores,
                on='SECID',
                how='inner',
            )

            if period_data.empty:
                continue

            # LOW_VOL_SCORE инвертирован, поэтому select_top_by_score
            # выбирает акции с наименьшей волатильностью
            selected = self.select_top_by_score(
                data=period_data,
                score_column='LOW_VOL_SCORE',
            )

            if selected.empty:
                continue

            frames.append(selected)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        """Финальная таблица с результатами стратегии."""
        period_returns = self.build_period_returns()

        if period_returns.empty:
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'LOW_VOL_SCORE',
            'VOLATILITY',
            'DATA_POINTS',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        available = [c for c in columns if c in period_returns.columns]

        return period_returns[available].sort_values(
            ['YEAR', 'LOW_VOL_SCORE'],
            ascending=[True, False],
        )




class AnomalyStrategyAnalyzer(StrategyAnalyzerBase):

    STRATEGY_NAME: str = 'Anomaly'
    ANOMALY_WINDOW: int = 60
    ANOMALY_THRESHOLD: float = 2.5
    MIN_DATA_POINTS: int = 90
    DEBUG: bool = False

    def _log(self, message: str) -> None:
        if self.DEBUG:
            print(f'[Anomaly] {message}')

    def load_daily_history(self) -> pd.DataFrame:
        if not DAILY_HISTORY_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Файл {DAILY_HISTORY_JSON_PATH} не найден. "
                "Запустите IssParser().parse_and_save() для его создания."
            )
        df = pd.read_json(DAILY_HISTORY_JSON_PATH)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        return df

    def _calculate_anomaly_scores(
        self,
        daily_prices: pd.DataFrame,
        signal_date: str,
    ) -> pd.DataFrame:
        signal_dt = pd.to_datetime(signal_date)
        self._log(f'Расчёт аномалий до {signal_dt}')

        historical = (
            daily_prices[daily_prices['TRADEDATE'] <= signal_dt]
            .copy()
            .sort_values(['SECID', 'TRADEDATE'])
        )

        if historical.empty:
            self._log('Нет исторических данных')
            return pd.DataFrame()

        self._log(f'Всего записей: {len(historical)}')
        self._log(f'Уникальных тикеров: {historical["SECID"].nunique()}')

        ticker_counts = historical.groupby('SECID')['TRADEDATE'].count()
        valid_tickers = ticker_counts[ticker_counts >= self.MIN_DATA_POINTS].index
        historical = historical[historical['SECID'].isin(valid_tickers)].copy()

        self._log(f'Тикеров с >= {self.MIN_DATA_POINTS} точек: {len(valid_tickers)}')

        if historical.empty:
            self._log('Нет тикеров с достаточным количеством данных')
            return pd.DataFrame()

        historical['LOG_RETURN'] = historical.groupby('SECID')['CLOSE'].transform(
            lambda x: np.log(x / x.shift(1))
        )

        def rolling_mad(series: pd.Series) -> pd.Series:
            rolling_median = series.rolling(
                self.ANOMALY_WINDOW, 
                min_periods=self.ANOMALY_WINDOW
            ).median()
            
            absolute_deviation = (series - rolling_median).abs()
            
            mad = absolute_deviation.rolling(
                self.ANOMALY_WINDOW, 
                min_periods=self.ANOMALY_WINDOW
            ).median()
            
            return mad

        historical['MAD'] = historical.groupby('SECID')['LOG_RETURN'].transform(
            rolling_mad
        )

        historical['ROLLING_MEDIAN'] = historical.groupby('SECID')['LOG_RETURN'].transform(
            lambda x: x.rolling(
                self.ANOMALY_WINDOW, 
                min_periods=self.ANOMALY_WINDOW
            ).median()
        )

        historical['MODIFIED_Z_SCORE'] = (
            0.6745 * (historical['LOG_RETURN'] - historical['ROLLING_MEDIAN']) 
            / historical['MAD']
        )

        last_scores = (
            historical.groupby('SECID')
            .last()
            .reset_index()
        )

        last_scores = last_scores[last_scores['MODIFIED_Z_SCORE'].notna()].copy()

        self._log(f'Тикеров с рассчитанным Z-Score: {len(last_scores)}')

        if last_scores.empty:
            self._log('Нет тикеров с рассчитанным Z-Score')
            return pd.DataFrame()

        if self.DEBUG and len(last_scores) > 0:
            self._log(f'Статистика Modified Z-Score:')
            self._log(f'  min={last_scores["MODIFIED_Z_SCORE"].min():.2f}, '
                     f'max={last_scores["MODIFIED_Z_SCORE"].max():.2f}, '
                     f'mean={last_scores["MODIFIED_Z_SCORE"].mean():.2f}')
            
            negative_anomalies = last_scores[last_scores['MODIFIED_Z_SCORE'] < -self.ANOMALY_THRESHOLD]
            self._log(f'Тикеров с отрицательной аномалией (Z < -{self.ANOMALY_THRESHOLD}): {len(negative_anomalies)}')

        last_scores['ANOMALY_SCORE'] = -last_scores['MODIFIED_Z_SCORE']

        result = last_scores[['SECID', 'ANOMALY_SCORE', 'MODIFIED_Z_SCORE']].copy()
        result['DATA_POINTS'] = (
            historical.groupby('SECID')['LOG_RETURN'].count().reindex(result['SECID']).values
        )

        return result

    def build_period_returns(self) -> pd.DataFrame:
        daily_prices = self.load_daily_history()
        all_returns = self.build_all_period_returns()

        if daily_prices.empty:
            self._log('Нет данных о ценах')
            return pd.DataFrame()
        
        if all_returns.empty:
            self._log('Нет данных о доходностях')
            return pd.DataFrame()

        frames = []

        for period in self.REBALANCE_PERIODS:
            self._log(f'Обработка периода {period.year}')

            anomaly_scores = self._calculate_anomaly_scores(
                daily_prices,
                signal_date=period.start_date,
            )

            if anomaly_scores.empty:
                self._log(f'Пустые аномалии для {period.year}')
                continue

            current_returns = all_returns[all_returns['YEAR'] == period.year].copy()
            if current_returns.empty:
                self._log(f'Пустые доходности для {period.year}')
                continue

            period_data = current_returns.merge(
                anomaly_scores,
                on='SECID',
                how='inner',
            )

            if period_data.empty:
                self._log(f'После merge нет данных для {period.year}')
                continue

            period_data_filtered = period_data[
                period_data['ANOMALY_SCORE'] > self.ANOMALY_THRESHOLD
            ].copy()

            self._log(f'Тикеров с аномалией > {self.ANOMALY_THRESHOLD}: {len(period_data_filtered)}')

            if period_data_filtered.empty:
                self._log(f'Нет аномалий выше порога для {period.year}, используем fallback (top-N без порога)')
                period_data_filtered = period_data.copy()

            selected = self.select_top_by_score(
                data=period_data_filtered,
                score_column='ANOMALY_SCORE',
            )

            if selected.empty:
                self._log(f'После select_top нет данных для {period.year}')
                continue

            self._log(f'Выбрано {len(selected)} акций для {period.year}')
            frames.append(selected)

        if not frames:
            self._log('Нет данных ни для одного периода')
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def analyze_holdings(self) -> pd.DataFrame:
        period_returns = self.build_period_returns()

        if period_returns.empty:
            self._log('analyze_holdings: пустой period_returns')
            return pd.DataFrame()

        columns = [
            'YEAR',
            'SECID',
            'SHORTNAME',
            'SECNAME',
            'LISTLEVEL',
            'ANOMALY_SCORE',
            'MODIFIED_Z_SCORE',
            'DATA_POINTS',
            'START_DATE',
            'END_DATE',
            'START_CLOSE',
            'END_CLOSE',
            'RETURN',
        ]

        available = [c for c in columns if c in period_returns.columns]

        return period_returns[available].sort_values(
            ['YEAR', 'ANOMALY_SCORE'],
            ascending=[True, False],
        )
    

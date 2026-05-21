from collections.abc import Iterable
from datetime import date, datetime
from random import uniform
from re import fullmatch, sub
from time import sleep

import pandas as pd
from bs4 import BeautifulSoup
from requests import get

from settings import DATA_DIR, HISTORY_JSON_PATH, MSFO_DATA_JSON_PATH, TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH


class IssParser:

    BASE_URL: str = 'https://iss.moex.com/iss'
    PAGE_MAX_SIZE: int = 100

    @staticmethod
    def wait() -> None:
        """Sleep for a random amount of time."""
        sleep(uniform(0.3, 0.9))  # noqa: S311

    def _get_page_table(self, url: str, table_name: str, params: dict | None = None) -> pd.DataFrame:

        params = {**(params or {}), 'iss.meta': 'off'}

        response = get(url, params=params, timeout=10)

        response.raise_for_status()

        payload = response.json()

        return pd.DataFrame(payload[table_name]['data'], columns=payload[table_name]['columns'])

    def get_page_table(self, url: str, table_name: str, params: dict | None = None) -> pd.DataFrame:

        params = params or {}

        page_tables = []
        start = 0

        while True:

            page = self._get_page_table(url, table_name, params={**params, 'start': start})

            if page.empty:
                break

            page_tables.append(page)

            if len(page) < self.PAGE_MAX_SIZE:
                break

            start += self.PAGE_MAX_SIZE

            self.wait()

        if not page_tables:
            return pd.DataFrame()

        return pd.concat(page_tables, ignore_index=True)

    def load_tqbr_top_listlevel_securities(self) -> pd.DataFrame:

        table = self._get_page_table(
            f'{self.BASE_URL}/engines/stock/markets/shares/boards/TQBR/securities.json',
            'securities',
            params={
                'securities.columns': ','.join([  # noqa: FLY002
                    'SECID',
                    'SHORTNAME',
                    'SECNAME',
                    'ISIN',
                    'LOTSIZE',
                    'LISTLEVEL',
                    'REGNUMBER',
                ])
            },
        )

        table = table[table['SECID'].notna()]

        return table[table['LISTLEVEL'].isin((1, 2))]

    def load_history_for_date(self, trading_date: date) -> pd.DataFrame:

        return self.get_page_table(
            f'{self.BASE_URL}/history/engines/stock/markets/shares/sessions/3/boards/TQBR/securities.json',
            'history',
            params={
                'date': trading_date.strftime('%Y-%m-%d'),
                'history.columns': ','.join([  # noqa: FLY002
                    'BOARDID',
                    'TRADEDATE',
                    'SECID',
                    'SHORTNAME',
                    'NUMTRADES',
                    'VALUE',
                    'OPEN',
                    'LOW',
                    'HIGH',
                    'WAPRICE',
                    'CLOSE',
                    'VOLUME',
                    'TRADINGSESSION',
                ]),
            },
        )

    def parse_and_save(self) -> None:

        DATA_DIR.mkdir(exist_ok=True)

        if not TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH.exists():
            self.load_tqbr_top_listlevel_securities().to_json(
                TQBR_TOP_LISTLEVEL_SECURITIES_JSON_PATH, force_ascii=False, indent=2
            )

        if not HISTORY_JSON_PATH.exists():
            pd.concat(
                [
                    self.load_history_for_date(date(year=2020, month=12, day=30)),
                    self.load_history_for_date(date(year=2021, month=12, day=30)),
                    self.load_history_for_date(date(year=2022, month=12, day=30)),
                    self.load_history_for_date(date(year=2023, month=12, day=29)),
                    self.load_history_for_date(date(year=2024, month=12, day=30)),
                    self.load_history_for_date(date(year=2025, month=12, day=30)),
                ],
                ignore_index=True,
            ).to_json(HISTORY_JSON_PATH, force_ascii=False, indent=2)


class SmartLabParser:

    BASE_URL: str = 'https://smart-lab.ru'

    @staticmethod
    def wait() -> None:
        """Sleep for a random amount of time."""
        sleep(uniform(0.7, 1.3))  # noqa: S311

    @staticmethod
    def normalize_text(value: str) -> str:
        return value.replace('\xa0', ' ').replace('\u200b', '').strip()

    def parse_number(self, value: str) -> float | None:

        value = self.normalize_text(value)

        if not value or value in ('-', '—'):
            return None

        is_percent = value.endswith('%')

        value = sub(r'[^0-9.\-]', '', value)

        if not value or value in ('-', '.', '-.'):
            return None

        try:
            number = float(value)

        except ValueError:
            return None

        if is_percent:
            return number / 100

        return number

    def parse_report_date(self, value: str) -> pd.Timestamp | None:

        value = self.normalize_text(value)

        if not value:
            return None

        try:
            return pd.Timestamp(datetime.strptime(value, '%d.%m.%Y').date())  # noqa: DTZ007

        except ValueError:
            pass

        return None

    def get_page_html(self, secid: str) -> str:

        response = get(
            f'{self.BASE_URL}/q/{secid}/f/y/MSFO/',
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                    ' Chrome/147.0.0.0 Safari/537.36 OPR/131.0.0.0'
                ),
                'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
            },
            timeout=10,
        )

        response.raise_for_status()

        return response.text

    def parse_page(self, secid: str) -> pd.DataFrame:  # noqa: PLR0912, C901

        soup = BeautifulSoup(self.get_page_html(secid), 'html.parser')

        rows = []

        for tr in soup.select_one('table.simple-little-table.financials').find_all('tr'):

            cells = tr.find_all(['th', 'td'])

            row = [self.normalize_text(cell.get_text(' ')) for cell in cells]

            if any(row):
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        max_len = max(len(row) for row in rows)
        rows = [row + [''] * (max_len - len(row)) for row in rows]

        raw_result = pd.DataFrame(rows)

        header_row_index = None

        for i, row in raw_result.iterrows():

            values = row.astype(str).tolist()
            years_amount = sum(bool(fullmatch(r'20\d{2}', self.normalize_text(value))) for value in values)

            if years_amount >= 2:  # noqa: PLR2004

                header_row_index = i
                break

        if header_row_index is None:
            raise ValueError(f'Could not find years header for {secid}')

        header = raw_result.iloc[header_row_index].tolist()[:-1]
        data = raw_result.iloc[header_row_index + 1:].copy()
        data = data.drop(columns=data.columns[1])
        data.columns = header

        data = data.rename(columns={data.columns[0]: 'METRIC'})

        year_columns = [column for column in data.columns if fullmatch(r'20\d{2}', str(column))]

        if not year_columns:
            return pd.DataFrame()

        result_rows = []
        report_dates = {}
        currencies = {}

        for _, row in data.iterrows():

            metric = self.normalize_text(str(row['METRIC']))

            if metric == 'Дата отчета':
                for year in year_columns:
                    report_dates[int(year)] = self.parse_report_date(str(row[year]))

            if metric == 'Валюта отчета':
                for year in year_columns:
                    currencies[int(year)] = self.normalize_text(str(row[year]))

        for _, row in data.iterrows():

            metric = self.normalize_text(str(row['METRIC']))

            if not metric:
                continue

            if metric in {'Дата отчета', 'Валюта отчета', 'Финансовый отчет', 'Годовой отчет', 'Презентация'}:
                continue

            for year in year_columns:

                year_int = int(year)
                value_raw = self.normalize_text(str(row[year]))

                result_rows.append({
                    'SECID': secid,
                    'YEAR': year_int,
                    'REPORT_DATE': report_dates.get(year_int),
                    'CURRENCY': currencies.get(year_int),
                    'METRIC': metric,
                    'VALUE_RAW': value_raw,
                    'VALUE': self.parse_number(value_raw),
                })

        return pd.DataFrame(result_rows)

    def parse_and_save(self, secids: Iterable[str]) -> None:

        DATA_DIR.mkdir(exist_ok=True)

        if MSFO_DATA_JSON_PATH.exists():
            return

        frames = []
        errors = []

        for secid in secids:

            try:

                frame = self.parse_page(secid=secid)

                if not frame.empty:
                    frames.append(frame)

            except Exception as e:  # noqa: BLE001
                errors.append({'SECID': secid, 'ERROR': str(e)})

            self.wait()

        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).to_json(
            MSFO_DATA_JSON_PATH, force_ascii=False, indent=2
        )

        errors_frame = pd.DataFrame(errors)

        if not errors_frame.empty:
            errors_frame.to_json(
                'data/smartlab_errors.json',
                orient='records',
                force_ascii=False,
                indent=2,
            )

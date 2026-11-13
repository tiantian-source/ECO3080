"""Build a CSI 500 stock-month panel and a separate financial-report audit table."""
import argparse
import math
from pathlib import Path
from zipfile import ZipFile
import pandas as pd


def build(raw, out, extra):
    def wide(name):
        return pd.read_pickle(raw / (name + '.pkl')).sort_index()

    def csvs(pattern, cols, strings=(), dates=(), keep=None):
        frames = []
        files = sorted(raw.glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        for path in files:
            with ZipFile(path) as z:
                for member in z.namelist():
                    if not member.lower().endswith('.csv'):
                        continue
                    with z.open(member) as stream:
                        for part in pd.read_csv(stream, usecols=cols, dtype={c: str for c in strings},
                                                parse_dates=list(dates), chunksize=250_000):
                            frames.append(part if keep is None else part.loc[keep(part)])
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    close = wide('daily_adj_close_post_20260630')
    calendar = close.index
    months = calendar.to_period('M')
    first = pd.Series(calendar, index=months).groupby(level=0).first()
    last = pd.Series(calendar, index=months).groupby(level=0).last()
    mask = wide('daily_csi500_components')
    formation = last.loc['2009-12':'2025-12']
    formation = formation[formation.isin(mask.index)]
    members = mask.loc[formation].stack()
    panel = members[members].rename('member').reset_index()
    panel.columns = ['formation_date', 'stock', 'member']
    panel = panel.drop(columns='member')
    if extra:
        addon = pd.read_csv(extra, parse_dates=['formation_date'], dtype={'stock': str})
        assert addon['formation_date'].isin(last.loc['2025-04':'2025-12']).all()
        assert addon['stock'].str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all()
        assert addon.groupby('formation_date')['stock'].nunique().eq(500).all()
        panel = pd.concat([panel, addon[['formation_date', 'stock']]], ignore_index=True)
    assert not panel.duplicated(['formation_date', 'stock']).any()
    panel['holding_month'] = panel['formation_date'].dt.to_period('M') + 1
    panel['entry_date'] = panel['holding_month'].map(first)
    panel['exit_date'] = (panel['holding_month'] + 1).map(first)
    panel['code'] = panel['stock'].str.split('.').str[0]
    stocks = pd.Index(panel.stock.unique())
    assert stocks.str.split('.').str[0].is_unique, 'Ambiguous six-digit stock code'

    def pick(data, dates):
        idx = pd.Index(dates)
        return data.reindex(index=idx.unique(), columns=stocks).to_numpy()[
            idx.unique().get_indexer(idx), stocks.get_indexer(panel.stock)]

    close = close.reindex(columns=stocks)
    monthly = close.reindex(last)
    monthly.index = last.index
    form = panel.formation_date.dt.to_period('M')
    panel['momentum'] = pick(monthly.shift(1) / monthly.shift(12) - 1, form)
    panel['reversal'] = 1 - pick(monthly / monthly.shift(1), form)
    panel['volatility'] = pick(close.pct_change(fill_method=None).rolling(21).std(), panel.formation_date)
    panel['entry_adj_close'] = pick(close, panel.entry_date)
    panel['exit_adj_close'] = pick(close, panel.exit_date)
    panel['stock_return'] = panel.exit_adj_close / panel.entry_adj_close - 1
    del close
    for col, name in [('market_cap', 'daily_a_share_market_val_3'), ('bm_unverified', 'daily_book_to_market_ratio_ttm')]:
        panel[col] = pick(wide(name + '_20260630'), panel.formation_date)
    panel['size'] = panel.market_cap.where(panel.market_cap.gt(0)).map(math.log, na_action='ignore')
    turnover = wide('daily_turnover_rate_20260630').reindex(index=calendar, columns=stocks)
    panel['turnover'] = pick(turnover.rolling(21).mean(), panel.formation_date)
    for col, name in [('ep_unverified', 'monthly_ep_ratio_ttm'), ('roa_unverified', 'monthly_return_on_asset_net_profit_ttm')]:
        data = wide(name + '_20260630')
        data.index = data.index.to_period('M')
        panel[col] = pick(data, form)
    for flag in ('suspend', 'st'):
        data = wide('stock_' + flag + '_flag_20260630')
        for side in ('entry', 'exit'):
            panel[side + '_' + flag] = pick(data, panel[side + '_date'])

    rf = csvs('*224152126*.zip', ['Clsdt', 'Nrrdaydt'], dates=['Clsdt']).set_index('Clsdt')['Nrrdaydt']
    rf = rf.reindex(pd.date_range(rf.index.min(), rf.index.max()))
    assert rf.notna().all(), 'Missing calendar-day risk-free rates'
    wealth = (1 + rf / 100).cumprod()
    panel['risk_free_return'] = panel.exit_date.map(wealth) / panel.entry_date.map(wealth) - 1
    assert panel['risk_free_return'].notna().all(), 'Risk-free coverage does not include every holding period'
    panel['excess_return'] = panel.stock_return - panel.risk_free_return

    codes = set(panel.code)
    days = set(panel.entry_date) | set(panel.exit_date)
    for pattern, code, day, cols in [
        ('日个股回报率文件*.zip', 'Stkcd', 'Trddt', ['LimitStatus']),
        ('融资融券标的信息表*.zip', 'Stockcode', 'Mtdate', ['Financeavailable', 'Shortavailable']),
    ]:
        daily = csvs(pattern, [code, day, *cols], [code], [day],
                     lambda x: x[code].isin(codes) & x[day].isin(days))
        daily = daily.rename(columns={code: 'code', day: 'date'}).set_index(['code', 'date'])
        assert daily.index.is_unique, f'Conflicting daily records: {pattern}'
        for side in ('entry', 'exit'):
            matched = daily.reindex(pd.MultiIndex.from_arrays([panel.code, panel[side + '_date']]))
            for col in cols:
                panel[f'{side}_{col}'] = matched[col].to_numpy()

    company = csvs('*225229154*.zip', ['Stkcd', 'ListedDate', 'DelistedDate'], ['Stkcd'],
                   ['ListedDate', 'DelistedDate']).rename(columns={'Stkcd': 'code'})
    ambiguous = company.loc[company.code.duplicated(keep=False), 'code'].unique()
    print('Ambiguous listing histories left unmatched:', ', '.join(sorted(set(ambiguous) & codes)) or 'none')
    company = company.loc[~company.code.isin(ambiguous)]
    panel = panel.merge(company, on='code', how='left', validate='many_to_one')
    keys = ['Stkcd', 'Accper', 'Typrep']
    assets = csvs('*224654030*.zip', keys + ['A001000000', 'A003100000'], ['Stkcd'], ['Accper'])
    income = csvs('*224903917*.zip', keys + ['B002000000', 'B002000101'], ['Stkcd'], ['Accper'])
    roa = csvs('*230338684*.zip', keys + ['Source', 'F050204C'], ['Stkcd'], ['Accper'])
    reports = assets.merge(income, on=keys, how='outer', validate='one_to_one')
    reports = reports.merge(roa, on=keys, how='outer', validate='one_to_one')
    announcements = csvs('*224353138*.zip', ['Stkcd', 'Accper', 'Reptyp', 'Annodt'], ['Stkcd'], ['Accper', 'Annodt'])
    reports = reports.merge(announcements, on=keys[:2], how='left', validate='many_to_one')
    # Vendor report values may be restated; announcement dates alone do not establish point-in-time values.
    out.mkdir(parents=True, exist_ok=True)
    panel.sort_values(['formation_date', 'stock']).to_csv(out / 'csi500_stock_month.csv', index=False)
    reports.loc[reports.Stkcd.isin(codes)].to_csv(out / 'financial_report_audit.csv', index=False)
    missing = last.loc['2009-12':'2025-12'].loc[lambda s: ~s.isin(panel.formation_date)]
    print(f'{len(panel):,} stock-month rows; last holding month: {panel.holding_month.max()}')
    print('Missing formation months:', ', '.join(missing.index.astype(str)) or 'none')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent)
    p.add_argument('--components', type=Path, help='Optional CSV: formation_date,stock (e.g. 600000.SH)')
    args = p.parse_args()
    build(args.input, args.output, args.components)

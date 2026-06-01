from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


NOTEBOOK = Path("notebook/correlation_tail_risk_sp_set_gold_btc.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Correlation + Tail Risk: S&P vs SET vs Gold vs BTC

            Notebook นี้เทียบความสัมพันธ์ของ 4 asset หลัก:

            - S&P 500 proxy: `SPY`
            - SET Index: `^SET.BK`
            - Gold proxy: `GC=F`
            - Bitcoin: `BTC-USD`

            วิเคราะห์ 2 ชุด:

            1. **Raw returns**: ผลตอบแทนรายวันของ asset โดยตรง
            2. **Daily exposure returns**: ผลตอบแทนหลังลด exposure ตาม risk trigger รายวัน

            มุมมองหลักเป็น **THB investor view**: USD assets ถูกคูณด้วย `USDTHB=X`
            เพื่อเทียบกับ SET Index ในหน่วย THB เดียวกัน
            """
        ),
        code(
            """
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import plotly.graph_objects as go
            import plotly.express as px

            ROOT = Path.cwd()
            if not (ROOT / 'data').exists() and (ROOT.parent / 'data').exists():
                ROOT = ROOT.parent
            OVERLAY_FILE = ROOT / 'data/cache/dynamic_factor_copula/overlay_compare_prices.parquet'
            EXTRA_FILE = ROOT / 'data/cache/dynamic_factor_copula/extra_prices.parquet'

            ASSET_ORDER = ['S&P 500', 'SET Index', 'Gold', 'BTC']
            TAIL_Q = 0.05
            DOWNSIDE_Q = 0.10

            print('Overlay file:', OVERLAY_FILE, OVERLAY_FILE.exists())
            print('Extra file:', EXTRA_FILE, EXTRA_FILE.exists())
            """
        ),
        md(
            """
            ## 1. Load And Align Data

            ใช้ local cache ที่มีอยู่ใน repo เพื่อให้ notebook reproducible และไม่ต้อง download ใหม่ทุกครั้ง
            """
        ),
        code(
            """
            overlay = pd.read_parquet(OVERLAY_FILE).sort_index().ffill()
            extra = pd.read_parquet(EXTRA_FILE, columns=['^SET.BK']).sort_index().ffill()

            prices_native = pd.concat(
                {
                    'SPY_USD': overlay['SPY'],
                    'SET_THB': extra['^SET.BK'],
                    'GOLD_USD': overlay['GC=F'],
                    'BTC_USD': overlay['BTC-USD'],
                    'VIX': overlay['^VIX'],
                    'USDTHB': overlay['USDTHB=X'],
                },
                axis=1,
            ).sort_index().ffill()

            prices_thb = pd.DataFrame(index=prices_native.index)
            prices_thb['S&P 500'] = prices_native['SPY_USD'] * prices_native['USDTHB']
            prices_thb['SET Index'] = prices_native['SET_THB']
            prices_thb['Gold'] = prices_native['GOLD_USD'] * prices_native['USDTHB']
            prices_thb['BTC'] = prices_native['BTC_USD'] * prices_native['USDTHB']
            prices_thb['VIX'] = prices_native['VIX']
            prices_thb = prices_thb.dropna(subset=ASSET_ORDER)

            raw_returns = (
                prices_thb[ASSET_ORDER]
                .pct_change(fill_method=None)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how='all')
            )

            pd.DataFrame({
                'Start': prices_thb[ASSET_ORDER].apply(lambda s: s.dropna().index.min().date()),
                'End': prices_thb[ASSET_ORDER].apply(lambda s: s.dropna().index.max().date()),
                'Observations': prices_thb[ASSET_ORDER].notna().sum(),
            })
            """
        ),
        md(
            """
            ## 2. Daily Exposure Rules

            - **S&P 500**: ลด exposure ถ้าต่ำกว่า MA200, drawdown ลึก, หรือ VIX สูง
            - **SET Index**: ลด exposure จาก MA200 และ drawdown ของ SET Index
            - **Gold**: ถ้าต่ำกว่า MA200 ลด exposure เหลือ 50%
            - **BTC**: ถ้าต่ำกว่า MA200 ลด exposure เหลือ 0%

            ส่วนที่ลด exposure ถือเป็น cash return 0% ต่อวัน
            """
        ),
        code(
            """
            def trend_drawdown_vix_exposure(price, vix=None, trend_cap=0.65, warn_cap=0.50, crash_cap=0.25):
                ma200 = price.rolling(200, min_periods=40).mean()
                drawdown = price / price.cummax() - 1.0
                candidates = [pd.Series(1.0, index=price.index, dtype=float)]
                candidates.append(pd.Series(np.where(price < ma200, trend_cap, 1.0), index=price.index))
                candidates.append(pd.Series(np.where(drawdown <= -0.08, warn_cap, 1.0), index=price.index))
                candidates.append(pd.Series(np.where(drawdown <= -0.15, crash_cap, 1.0), index=price.index))
                if vix is not None:
                    vix_aligned = vix.reindex(price.index).ffill()
                    candidates.append(pd.Series(np.where(vix_aligned >= 28.0, warn_cap, 1.0), index=price.index))
                    candidates.append(pd.Series(np.where(vix_aligned >= 35.0, crash_cap, 1.0), index=price.index))
                exposure = pd.concat(candidates, axis=1).min(axis=1).clip(0.0, 1.0)
                exposure.loc[ma200.isna()] = 1.0
                return exposure

            def ma_exposure(price, below=0.50):
                ma200 = price.rolling(200, min_periods=40).mean()
                exposure = pd.Series(1.0, index=price.index, dtype=float)
                exposure.loc[price < ma200] = below
                exposure.loc[ma200.isna()] = 1.0
                return exposure.clip(0.0, 1.0)

            exposure = pd.DataFrame(index=prices_thb.index)
            exposure['S&P 500'] = trend_drawdown_vix_exposure(prices_thb['S&P 500'], prices_thb['VIX'])
            exposure['SET Index'] = trend_drawdown_vix_exposure(prices_thb['SET Index'])
            exposure['Gold'] = ma_exposure(prices_thb['Gold'], below=0.50)
            exposure['BTC'] = ma_exposure(prices_thb['BTC'], below=0.00)

            lagged_exposure = exposure.shift(1).reindex(raw_returns.index).ffill().fillna(1.0)
            exposure_returns = raw_returns.mul(lagged_exposure, axis=0)
            exposure.describe().T[['mean', 'min', '25%', '50%', '75%', 'max']]
            """
        ),
        code(
            """
            fig = go.Figure()
            for asset in ASSET_ORDER:
                fig.add_trace(go.Scatter(x=exposure.index, y=exposure[asset], mode='lines', name=asset))
            fig.update_layout(
                title='Daily Exposure Path By Asset',
                xaxis_title='Date',
                yaxis_title='Exposure',
                yaxis_tickformat='.0%',
                height=480,
            )
            fig.show()
            """
        ),
        md(
            """
            ## 3. Correlation And Tail-Risk Functions

            - **Pearson correlation**: linear correlation across all daily returns
            - **Spearman correlation**: rank correlation
            - **Downside correlation**: correlation on days where either asset is in its bottom 10% daily returns
            - **Lower-tail dependence**: probability asset B is also in bottom 5% when asset A is in bottom 5%
            """
        ),
        code(
            """
            def downside_corr_matrix(returns, q=0.10):
                out = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
                thresholds = returns.quantile(q)
                for a in returns.columns:
                    for b in returns.columns:
                        if a == b:
                            out.loc[a, b] = 1.0
                            continue
                        mask = (returns[a] <= thresholds[a]) | (returns[b] <= thresholds[b])
                        sample = returns.loc[mask, [a, b]].dropna()
                        out.loc[a, b] = sample[a].corr(sample[b]) if len(sample) >= 10 else np.nan
                return out

            def lower_tail_dependence_matrix(returns, q=0.05):
                out = pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
                thresholds = returns.quantile(q)
                tail = returns.le(thresholds, axis=1)
                for a in returns.columns:
                    denom = tail[a].sum()
                    for b in returns.columns:
                        out.loc[a, b] = (tail[a] & tail[b]).sum() / denom if denom else np.nan
                return out

            def summarize_pair_tail_metrics(returns, label):
                pearson = returns.corr(method='pearson')
                spearman = returns.corr(method='spearman')
                down = downside_corr_matrix(returns, q=DOWNSIDE_Q)
                tail_dep = lower_tail_dependence_matrix(returns, q=TAIL_Q)
                rows = []
                for i, a in enumerate(returns.columns):
                    for b in returns.columns[i+1:]:
                        rows.append({
                            'Return Set': label,
                            'Pair': f'{a} vs {b}',
                            'Pearson': pearson.loc[a, b],
                            'Spearman': spearman.loc[a, b],
                            f'Downside Corr q{int(DOWNSIDE_Q*100)}': down.loc[a, b],
                            f'Tail Dep {a}-> {b} q{int(TAIL_Q*100)}': tail_dep.loc[a, b],
                            f'Tail Dep {b}-> {a} q{int(TAIL_Q*100)}': tail_dep.loc[b, a],
                        })
                return pd.DataFrame(rows)

            raw_pair_summary = summarize_pair_tail_metrics(raw_returns[ASSET_ORDER].dropna(), 'Raw returns')
            exposure_pair_summary = summarize_pair_tail_metrics(exposure_returns[ASSET_ORDER].dropna(), 'Daily exposure returns')
            pair_summary = pd.concat([raw_pair_summary, exposure_pair_summary], ignore_index=True)
            pair_summary.sort_values(['Return Set', f'Downside Corr q{int(DOWNSIDE_Q*100)}'], ascending=[True, False])
            """
        ),
        md("## 4. Heatmaps"),
        code(
            """
            def heatmap(matrix, title, zmin=-1, zmax=1, colorscale='RdBu'):
                fig = px.imshow(
                    matrix.astype(float),
                    text_auto='.2f',
                    color_continuous_scale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    aspect='auto',
                    title=title,
                )
                fig.update_layout(height=460)
                fig.show()

            heatmap(raw_returns[ASSET_ORDER].dropna().corr(), 'Raw Returns: Pearson Correlation')
            heatmap(exposure_returns[ASSET_ORDER].dropna().corr(), 'Daily Exposure Returns: Pearson Correlation')
            """
        ),
        code(
            """
            raw_downside = downside_corr_matrix(raw_returns[ASSET_ORDER].dropna(), q=DOWNSIDE_Q)
            exp_downside = downside_corr_matrix(exposure_returns[ASSET_ORDER].dropna(), q=DOWNSIDE_Q)
            heatmap(raw_downside, f'Raw Returns: Downside Correlation, Bottom {int(DOWNSIDE_Q*100)}% Days')
            heatmap(exp_downside, f'Daily Exposure Returns: Downside Correlation, Bottom {int(DOWNSIDE_Q*100)}% Days')
            """
        ),
        code(
            """
            raw_tail_dep = lower_tail_dependence_matrix(raw_returns[ASSET_ORDER].dropna(), q=TAIL_Q)
            exp_tail_dep = lower_tail_dependence_matrix(exposure_returns[ASSET_ORDER].dropna(), q=TAIL_Q)
            heatmap(raw_tail_dep, f'Raw Returns: Lower-Tail Dependence, Bottom {int(TAIL_Q*100)}%', zmin=0, zmax=1, colorscale='Reds')
            heatmap(exp_tail_dep, f'Daily Exposure Returns: Lower-Tail Dependence, Bottom {int(TAIL_Q*100)}%', zmin=0, zmax=1, colorscale='Reds')
            """
        ),
        md(
            """
            ## 5. Compare Raw vs Daily Exposure

            ตารางนี้ดูว่า daily exposure ทำให้ correlation และ tail dependence ของแต่ละ pair เปลี่ยนอย่างไร
            """
        ),
        code(
            """
            raw_named = raw_pair_summary.set_index('Pair')
            exp_named = exposure_pair_summary.set_index('Pair')
            compare = pd.DataFrame(index=raw_named.index)
            for col in ['Pearson', 'Spearman', f'Downside Corr q{int(DOWNSIDE_Q*100)}']:
                compare[f'Raw {col}'] = raw_named[col]
                compare[f'Exposure {col}'] = exp_named[col]
                compare[f'Change {col}'] = exp_named[col] - raw_named[col]
            compare.sort_values(f'Raw Downside Corr q{int(DOWNSIDE_Q*100)}', ascending=False)
            """
        ),
        code(
            """
            bar_data = pair_summary.melt(
                id_vars=['Return Set', 'Pair'],
                value_vars=['Pearson', f'Downside Corr q{int(DOWNSIDE_Q*100)}'],
                var_name='Metric',
                value_name='Value',
            )
            fig = px.bar(
                bar_data,
                x='Pair',
                y='Value',
                color='Return Set',
                facet_col='Metric',
                barmode='group',
                title='Pair Correlation: Raw vs Daily Exposure',
            )
            fig.update_layout(height=520, yaxis_title='Correlation')
            fig.show()
            """
        ),
        md(
            """
            ## 6. Interpretation Checklist

            อ่านผลโดยดู 3 ชั้น:

            1. **Pearson/Spearman**: ความสัมพันธ์ปกติทั้งช่วงเวลา
            2. **Downside correlation**: ความสัมพันธ์ในวันที่อย่างน้อยหนึ่ง asset กำลังอยู่ในช่วงแย่
            3. **Lower-tail dependence**: ถ้า asset หนึ่งแย่ระดับ bottom 5%, อีก asset มีโอกาสแย่พร้อมกันกี่ %

            ถ้า daily exposure ลด downside correlation หรือ lower-tail dependence ลง
            แปลว่า risk trigger ช่วยลดการโดน tail-risk พร้อมกันระหว่าง asset ได้บางส่วน
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    nbf.write(nb, NOTEBOOK)
    print(f"Wrote executed notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()

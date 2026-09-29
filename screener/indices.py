"""Index registry and constituent fetching.

Indian indices come from the official constituent CSVs published by NSE Indices,
the S&P 500 from the constituents table on Wikipedia and the Nasdaq-100 from
Nasdaq's public quote API.
Tickers are returned in Yahoo Finance notation (``RELIANCE.NS``, ``BRK-B``).
"""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass

import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
HTTP_TIMEOUT = 30
CONSTITUENTS_TTL_SECONDS = 24 * 60 * 60

NSE_CSV_HOSTS = (
    "https://www.niftyindices.com/IndexConstituent/{file}",
    "https://nsearchives.nseindia.com/content/indices/{file}",
)


class UnknownIndexError(ValueError):
    pass


class ConstituentsFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexSpec:
    key: str
    name: str
    market: str  # "IN" | "US"
    source: str  # "nse_csv" | "wikipedia" | "nasdaq_api"
    location: str  # CSV file name or URL
    ticker_suffix: str = ""


@dataclass(frozen=True)
class Constituent:
    symbol: str  # exchange symbol, e.g. RELIANCE / BRK.B
    ticker: str  # Yahoo Finance ticker, e.g. RELIANCE.NS / BRK-B
    name: str
    sector: str | None = None


def _nse(key: str, name: str, file: str) -> IndexSpec:
    return IndexSpec(key, name, "IN", "nse_csv", file, ".NS")


def _wiki(key: str, name: str, url: str) -> IndexSpec:
    return IndexSpec(key, name, "US", "wikipedia", url)


_SPECS = [
    _nse("NIFTY50", "Nifty 50", "ind_nifty50list.csv"),
    _nse("NIFTYNEXT50", "Nifty Next 50", "ind_niftynext50list.csv"),
    _nse("NIFTY100", "Nifty 100", "ind_nifty100list.csv"),
    _nse("NIFTY200", "Nifty 200", "ind_nifty200list.csv"),
    _nse("NIFTY500", "Nifty 500", "ind_nifty500list.csv"),
    _nse("NIFTYMIDCAP50", "Nifty Midcap 50", "ind_niftymidcap50list.csv"),
    _nse("NIFTYMIDCAP100", "Nifty Midcap 100", "ind_niftymidcap100list.csv"),
    _nse("NIFTYMIDCAP150", "Nifty Midcap 150", "ind_niftymidcap150list.csv"),
    _nse("NIFTYSMALLCAP50", "Nifty Smallcap 50", "ind_niftysmallcap50list.csv"),
    _nse("NIFTYSMALLCAP100", "Nifty Smallcap 100", "ind_niftysmallcap100list.csv"),
    _nse("NIFTYSMALLCAP250", "Nifty Smallcap 250", "ind_niftysmallcap250list.csv"),
    _nse("NIFTYBANK", "Nifty Bank", "ind_niftybanklist.csv"),
    _nse("NIFTYIT", "Nifty IT", "ind_niftyitlist.csv"),
    _nse("NIFTYAUTO", "Nifty Auto", "ind_niftyautolist.csv"),
    _nse("NIFTYPHARMA", "Nifty Pharma", "ind_niftypharmalist.csv"),
    _nse("NIFTYFMCG", "Nifty FMCG", "ind_niftyfmcglist.csv"),
    _nse("NIFTYMETAL", "Nifty Metal", "ind_niftymetallist.csv"),
    _wiki("SP500", "S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    IndexSpec(
        "NASDAQ100", "Nasdaq-100", "US", "nasdaq_api",
        "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
    ),
]

INDICES: dict[str, IndexSpec] = {spec.key: spec for spec in _SPECS}

_ALIASES = {
    "NIFTY": "NIFTY50",
    "NIFTYJUNIOR": "NIFTYNEXT50",
    "BANKNIFTY": "NIFTYBANK",
    "CNX500": "NIFTY500",
    "MIDCAP100": "NIFTYMIDCAP100",
    "SMALLCAP100": "NIFTYSMALLCAP100",
    "SPX": "SP500",
    "GSPC": "SP500",
    "SANDP500": "SP500",
    "NDX": "NASDAQ100",
    "NASDAQ": "NASDAQ100",
}


def resolve_index(name: str) -> IndexSpec:
    """Map user input such as ``"nifty 50"``, ``"S&P 500"`` or ``"^NDX"`` to a spec."""
    key = re.sub(r"[^A-Z0-9]", "", name.upper().replace("&", ""))
    key = _ALIASES.get(key, key)
    try:
        return INDICES[key]
    except KeyError:
        raise UnknownIndexError(
            f"Unknown index '{name}'. Supported: {', '.join(INDICES)}"
        ) from None


def _http_get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def parse_nse_csv(text: str, suffix: str = ".NS") -> list[Constituent]:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    if "Symbol" not in df.columns:
        raise ConstituentsFetchError("NSE constituent CSV has no 'Symbol' column")
    out = []
    for row in df.to_dict("records"):
        symbol = str(row["Symbol"]).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        sector = row.get("Industry")
        out.append(
            Constituent(
                symbol=symbol,
                ticker=f"{symbol}{suffix}",
                name=str(row.get("Company Name", symbol)).strip(),
                sector=str(sector).strip() if pd.notna(sector) else None,
            )
        )
    return out


_SYMBOL_COLUMNS = ("symbol", "ticker", "ticker symbol")
_NAME_COLUMNS = ("security", "company", "company name")
_SECTOR_COLUMNS = ("gics sector", "icb industry", "sector", "industry")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(col[-1]) for col in df.columns]
    return df


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    # Wikipedia headers often carry footnote markers, e.g. "Symbol[a]".
    lowered = {re.sub(r"\[.*?\]", "", str(c)).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_wikipedia_constituents(html: str, suffix: str = "") -> list[Constituent]:
    """Pick the constituents table: the largest one that has a ticker and a company column."""
    best: tuple[pd.DataFrame, str, str] | None = None
    for table in pd.read_html(io.StringIO(html)):
        table = _flatten_columns(table)
        symbol_col = _pick_column(table, _SYMBOL_COLUMNS)
        name_col = _pick_column(table, _NAME_COLUMNS)
        if symbol_col is None or name_col is None:
            continue
        if best is None or len(table) > len(best[0]):
            best = (table, symbol_col, name_col)
    if best is None:
        raise ConstituentsFetchError("No constituents table found on the Wikipedia page")

    table, symbol_col, name_col = best
    sector_col = _pick_column(table, _SECTOR_COLUMNS)
    out = []
    for row in table.to_dict("records"):
        symbol = str(row[symbol_col]).strip().upper()
        # Guard against footnote / header rows that sneak into the table body.
        if not re.fullmatch(r"[A-Z0-9.\-]{1,10}", symbol):
            continue
        sector = row.get(sector_col) if sector_col else None
        out.append(
            Constituent(
                symbol=symbol,
                ticker=symbol.replace(".", "-") + suffix,  # BRK.B -> BRK-B on Yahoo
                name=str(row[name_col]).strip(),
                sector=str(sector).strip() if sector is not None and pd.notna(sector) else None,
            )
        )
    return out


def parse_nasdaq_api(payload: dict, suffix: str = "") -> list[Constituent]:
    try:
        rows = payload["data"]["data"]["rows"]
    except (KeyError, TypeError):
        raise ConstituentsFetchError("Unexpected response shape from the Nasdaq API") from None
    out = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        # "Apple Inc. Common Stock" -> "Apple Inc."
        name = re.sub(
            r"\s+(Common Stock|Common Shares|Ordinary Shares|Class [A-C]\b|American Depositary).*$",
            "",
            str(row.get("companyName", symbol)).strip(),
        )
        out.append(
            Constituent(
                symbol=symbol,
                ticker=symbol.replace(".", "-").replace("/", "-") + suffix,
                name=name,
                sector=str(row.get("sector") or "").strip() or None,
            )
        )
    return out


def _fetch(spec: IndexSpec) -> list[Constituent]:
    if spec.source == "nse_csv":
        errors = []
        for template in NSE_CSV_HOSTS:
            url = template.format(file=spec.location)
            try:
                return parse_nse_csv(_http_get(url), spec.ticker_suffix)
            except (requests.RequestException, ConstituentsFetchError, pd.errors.ParserError) as exc:
                errors.append(f"{url}: {exc}")
        raise ConstituentsFetchError("; ".join(errors))
    try:
        if spec.source == "nasdaq_api":
            resp = requests.get(
                spec.location,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return parse_nasdaq_api(resp.json(), spec.ticker_suffix)
        return parse_wikipedia_constituents(_http_get(spec.location), spec.ticker_suffix)
    except (requests.RequestException, ValueError) as exc:
        raise ConstituentsFetchError(f"{spec.location}: {exc}") from exc


_cache: dict[str, tuple[float, list[Constituent]]] = {}
_cache_lock = threading.Lock()


def get_constituents(spec: IndexSpec, refresh: bool = False) -> list[Constituent]:
    with _cache_lock:
        hit = _cache.get(spec.key)
        if hit and not refresh and time.time() - hit[0] < CONSTITUENTS_TTL_SECONDS:
            return hit[1]
    constituents = _fetch(spec)
    if not constituents:
        raise ConstituentsFetchError(f"No constituents found for {spec.key}")
    with _cache_lock:
        _cache[spec.key] = (time.time(), constituents)
    return constituents

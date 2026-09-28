"""DuckDB load/query helpers: a fast, file-based, server-less analytical store for
normalized sales/product/context data. Adapters write here once loaded; the tools
layer reads from here rather than re-parsing raw sources on every call.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.schema.core import MarketContext, Product, SalesObservation

_PRODUCTS_TABLE = "products"
_SALES_TABLE = "sales_observations"
_CONTEXT_TABLE = "market_context"


class DuckDBStore:
    """Thin wrapper around a DuckDB file: normalized Pydantic models in, pandas/typed
    query results out. One store per `settings.duckdb_path` (see `config/settings.py`).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write_products(self, products: list[Product]) -> None:
        if not products:
            return
        df = pd.DataFrame([p.model_dump() for p in products])
        df["attributes"] = df["attributes"].apply(str)  # duckdb has no native dict type
        self._conn.register("_products_df", df)
        self._conn.execute(
            f"CREATE OR REPLACE TABLE {_PRODUCTS_TABLE} AS SELECT * FROM _products_df"
        )
        self._conn.unregister("_products_df")

    def write_sales(self, observations: list[SalesObservation]) -> None:
        if not observations:
            return
        df = pd.DataFrame([o.model_dump() for o in observations])
        self._conn.register("_sales_df", df)
        self._conn.execute(f"CREATE OR REPLACE TABLE {_SALES_TABLE} AS SELECT * FROM _sales_df")
        self._conn.unregister("_sales_df")

    def write_market_context(self, contexts: list[MarketContext]) -> None:
        if not contexts:
            return
        df = pd.DataFrame([c.model_dump() for c in contexts])
        df["macro_index"] = df["macro_index"].apply(str)
        df["competitor_pressure"] = df["competitor_pressure"].apply(str)
        self._conn.register("_context_df", df)
        self._conn.execute(f"CREATE OR REPLACE TABLE {_CONTEXT_TABLE} AS SELECT * FROM _context_df")
        self._conn.unregister("_context_df")

    def query(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        return self._conn.execute(sql, params or []).fetchdf()

    def sales_for_product(self, product_id: str) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {_SALES_TABLE} WHERE product_id = ? ORDER BY period",  # noqa: S608
            [product_id],
        )

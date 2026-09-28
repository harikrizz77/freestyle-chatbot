from datetime import date
from pathlib import Path

from src.data.store import DuckDBStore
from src.schema.core import MarketContext, Product, SalesObservation


def test_write_and_query_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "store.duckdb"
    products = [
        Product(
            product_id="A1", name="A1", category="c", brand="b", price_tier="mid", is_focal=True
        )
    ]
    sales = [
        SalesObservation(
            product_id="A1", region="US", period=date(2024, 1, 1), units=10, price=5.0
        ),
        SalesObservation(
            product_id="A1", region="US", period=date(2024, 1, 8), units=12, price=5.0
        ),
    ]
    contexts = [MarketContext(region="US", period=date(2024, 1, 1), macro_index={"cpi": 300.0})]

    with DuckDBStore(db_path) as store:
        store.write_products(products)
        store.write_sales(sales)
        store.write_market_context(contexts)

        df = store.sales_for_product("A1")
        assert len(df) == 2
        assert df["units"].sum() == 22

        products_df = store.query("SELECT * FROM products")
        assert len(products_df) == 1

    assert db_path.exists()


def test_write_empty_lists_are_noops(tmp_path: Path) -> None:
    with DuckDBStore(tmp_path / "store.duckdb") as store:
        store.write_products([])
        store.write_sales([])
        store.write_market_context([])
        # no tables created, but no error raised either

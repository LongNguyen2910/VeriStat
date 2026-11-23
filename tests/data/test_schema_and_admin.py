import sqlite3
import unicodedata
from pathlib import Path
import pytest
from veristat.data.schema import init_database, open_readonly
from veristat.data.admin_mapping import (
    classify_and_normalize_geo,
    normalize_entity,
    normalize_entity_list,
    populate_admin_mapping,
)


def test_readonly_and_fts5_tables(tmp_path: Path):
    db_path = tmp_path / "veristat.db"
    conn = init_database(db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert {"tables_catalog", "admin_mapping", "docs_glossary"}.issubset(names)
    conn.close()
    ro = open_readonly(db_path)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("CREATE TABLE hack (id INT)")
    ro.close()


def test_geo_level_separation_and_vietnamese_normalization():
    assert classify_and_normalize_geo("CẢ NƯỚC") == ("Cả nước", "country")
    assert classify_and_normalize_geo("Đồng bằng sông Hồng") == ("Đồng bằng sông Hồng", "region")
    nfd = unicodedata.normalize("NFD", "thành phố Hoà Bình")
    assert classify_and_normalize_geo(nfd) == ("Hòa Bình", "province")
    assert normalize_entity("TP.HCM") == "TP. Hồ Chí Minh"
    assert normalize_entity("Hà Tây") == "Hà Nội"
    assert normalize_entity_list(["TP.HCM", "Hà Tây"]) == ["TP. Hồ Chí Minh", "Hà Nội"]


def test_admin_mapping_covers_audit_entities_and_is_idempotent(tmp_path: Path):
    conn = init_database(tmp_path / "veristat.db")
    n1 = populate_admin_mapping(conn)
    n2 = populate_admin_mapping(conn)
    assert n1 >= 63 and n1 == n2
    conn.close()

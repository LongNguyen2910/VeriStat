import math
from pathlib import Path
from veristat.data.benchmark_gen import (
    check_answer_match,
    has_ambiguous_topk_tie,
    execute_reference_code,
)
from veristat.data.schema import init_database


def test_check_answer_match_all_types():
    assert check_answer_match(100.05, 100.0, "number", rel_tol=1e-3) is True
    assert check_answer_match(100.2, 100.0, "number", rel_tol=1e-3) is False
    assert check_answer_match("TP.HCM", "TP. Hồ Chí Minh", "entity") is True
    assert check_answer_match(["TP.HCM", "Hà Tây"], ["TP. Hồ Chí Minh", "Hà Nội"], "entity_list") is True
    assert check_answer_match(["Hà Nội", "TP. Hồ Chí Minh"], ["TP. Hồ Chí Minh", "Hà Nội"], "entity_list") is False


def test_has_ambiguous_topk_tie():
    assert has_ambiguous_topk_tie([10.0, 8.0, 8.0, 5.0], k=2) is True
    assert has_ambiguous_topk_tie([10.0, 8.0, 7.0, 5.0], k=2) is False


def test_execute_reference_code_readonly(tmp_path: Path):
    db_path = tmp_path / "veristat.db"
    conn = init_database(db_path)
    conn.execute("CREATE TABLE t (x REAL)")
    conn.execute("INSERT INTO t VALUES (10.0), (20.0)")
    conn.commit()
    conn.close()
    val = execute_reference_code(db_path, "rows = [r[0] for r in conn.execute('SELECT x FROM t')]; result = sum(rows)/len(rows)")
    assert math.isclose(val, 15.0, rel_tol=1e-9)

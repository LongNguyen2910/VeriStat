"""Audit-driven administrative boundary mapping and Vietnamese geo normalization.

Built directly from Phase 1 Data Audit findings (`docs/data_audit.json`):
- NG-TOAN-QUOC-2025-1.pdf: CẢ NƯỚC + 6 socio-economic regions + 63 provinces
  (with Thừa Thiên Huế already renamed to Huế effective 2025-01-01).
- 2.-Bieu-thang-10.2026.xlsx: CẢ NƯỚC + 34 provinces after the 2025 Vietnam
  provincial merger (Resolution 202/2025/QH15, effective 2025-07-01).
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from veristat.data.schema import GeoLevel

# Old-style -> new-style Vietnamese tone placement replacements.
_OLD_TO_NEW_TONE: tuple[tuple[str, str], ...] = (
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("Oà", "Òa"), ("Oá", "Óa"), ("Oả", "Ỏa"), ("Oã", "Õa"), ("Oạ", "Ọa"),
    ("OÀ", "ÒA"), ("OÁ", "ÓA"), ("OẢ", "ỎA"), ("OÃ", "ÕA"), ("OẠ", "ỌA"),
    ("oè", "òe"), ("oé", "óe"), ("oẻ", "ỏe"), ("oẽ", "õe"), ("oẹ", "ọe"),
    ("Oè", "Òe"), ("Oé", "Óe"), ("Oẻ", "Ỏe"), ("Oẽ", "Õe"), ("Oẹ", "Ọe"),
    ("OÈ", "ÒE"), ("OÉ", "ÓE"), ("OẺ", "ỎE"), ("OẼ", "ÕE"), ("OẸ", "ỌE"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
    ("Uỳ", "Ùy"), ("Uý", "Úy"), ("Uỷ", "Ủy"), ("Uỹ", "Ũy"), ("Uỵ", "Ụy"),
    ("UỲ", "ÙY"), ("UÝ", "ÚY"), ("UỶ", "ỦY"), ("UỸ", "ŨY"), ("UỴ", "ỤY"),
)

# 6 socio-economic regions of Vietnam (canonical names from NG-TOAN-QUOC-2025-1.pdf).
REGIONS: tuple[str, ...] = (
    "Đồng bằng sông Hồng",
    "Trung du và miền núi phía Bắc",
    "Bắc Trung Bộ và Duyên hải miền Trung",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
)

# 63 provinces in NG-TOAN-QUOC-2025-1.pdf -> (region_name, canonical_2026_after_merger).
PROVINCES_63: dict[str, tuple[str, str]] = {
    # 1. Đồng bằng sông Hồng (11)
    "Hà Nội": ("Đồng bằng sông Hồng", "Hà Nội"),
    "Vĩnh Phúc": ("Đồng bằng sông Hồng", "Phú Thọ"),
    "Bắc Ninh": ("Đồng bằng sông Hồng", "Bắc Ninh"),
    "Quảng Ninh": ("Đồng bằng sông Hồng", "Quảng Ninh"),
    "Hải Dương": ("Đồng bằng sông Hồng", "Hải Phòng"),
    "Hải Phòng": ("Đồng bằng sông Hồng", "Hải Phòng"),
    "Hưng Yên": ("Đồng bằng sông Hồng", "Hưng Yên"),
    "Thái Bình": ("Đồng bằng sông Hồng", "Hưng Yên"),
    "Hà Nam": ("Đồng bằng sông Hồng", "Ninh Bình"),
    "Nam Định": ("Đồng bằng sông Hồng", "Ninh Bình"),
    "Ninh Bình": ("Đồng bằng sông Hồng", "Ninh Bình"),
    # 2. Trung du và miền núi phía Bắc (14)
    "Hà Giang": ("Trung du và miền núi phía Bắc", "Tuyên Quang"),
    "Cao Bằng": ("Trung du và miền núi phía Bắc", "Cao Bằng"),
    "Bắc Kạn": ("Trung du và miền núi phía Bắc", "Thái Nguyên"),
    "Tuyên Quang": ("Trung du và miền núi phía Bắc", "Tuyên Quang"),
    "Lào Cai": ("Trung du và miền núi phía Bắc", "Lào Cai"),
    "Yên Bái": ("Trung du và miền núi phía Bắc", "Lào Cai"),
    "Thái Nguyên": ("Trung du và miền núi phía Bắc", "Thái Nguyên"),
    "Lạng Sơn": ("Trung du và miền núi phía Bắc", "Lạng Sơn"),
    "Bắc Giang": ("Trung du và miền núi phía Bắc", "Bắc Ninh"),
    "Phú Thọ": ("Trung du và miền núi phía Bắc", "Phú Thọ"),
    "Điện Biên": ("Trung du và miền núi phía Bắc", "Điện Biên"),
    "Lai Châu": ("Trung du và miền núi phía Bắc", "Lai Châu"),
    "Sơn La": ("Trung du và miền núi phía Bắc", "Sơn La"),
    "Hòa Bình": ("Trung du và miền núi phía Bắc", "Phú Thọ"),
    # 3. Bắc Trung Bộ và Duyên hải miền Trung (14)
    "Thanh Hóa": ("Bắc Trung Bộ và Duyên hải miền Trung", "Thanh Hóa"),
    "Nghệ An": ("Bắc Trung Bộ và Duyên hải miền Trung", "Nghệ An"),
    "Hà Tĩnh": ("Bắc Trung Bộ và Duyên hải miền Trung", "Hà Tĩnh"),
    "Quảng Bình": ("Bắc Trung Bộ và Duyên hải miền Trung", "Quảng Trị"),
    "Quảng Trị": ("Bắc Trung Bộ và Duyên hải miền Trung", "Quảng Trị"),
    "Huế": ("Bắc Trung Bộ và Duyên hải miền Trung", "Huế"),
    "Đà Nẵng": ("Bắc Trung Bộ và Duyên hải miền Trung", "Đà Nẵng"),
    "Quảng Nam": ("Bắc Trung Bộ và Duyên hải miền Trung", "Đà Nẵng"),
    "Quảng Ngãi": ("Bắc Trung Bộ và Duyên hải miền Trung", "Quảng Ngãi"),
    "Bình Định": ("Bắc Trung Bộ và Duyên hải miền Trung", "Gia Lai"),
    "Phú Yên": ("Bắc Trung Bộ và Duyên hải miền Trung", "Đắk Lắk"),
    "Khánh Hòa": ("Bắc Trung Bộ và Duyên hải miền Trung", "Khánh Hòa"),
    "Ninh Thuận": ("Bắc Trung Bộ và Duyên hải miền Trung", "Khánh Hòa"),
    "Bình Thuận": ("Bắc Trung Bộ và Duyên hải miền Trung", "Lâm Đồng"),
    # 4. Tây Nguyên (5)
    "Kon Tum": ("Tây Nguyên", "Quảng Ngãi"),
    "Gia Lai": ("Tây Nguyên", "Gia Lai"),
    "Đắk Lắk": ("Tây Nguyên", "Đắk Lắk"),
    "Đắk Nông": ("Tây Nguyên", "Lâm Đồng"),
    "Lâm Đồng": ("Tây Nguyên", "Lâm Đồng"),
    # 5. Đông Nam Bộ (6)
    "Bình Phước": ("Đông Nam Bộ", "Đồng Nai"),
    "Tây Ninh": ("Đông Nam Bộ", "Tây Ninh"),
    "Bình Dương": ("Đông Nam Bộ", "TP. Hồ Chí Minh"),
    "Đồng Nai": ("Đông Nam Bộ", "Đồng Nai"),
    "Bà Rịa - Vũng Tàu": ("Đông Nam Bộ", "TP. Hồ Chí Minh"),
    "TP. Hồ Chí Minh": ("Đông Nam Bộ", "TP. Hồ Chí Minh"),
    # 6. Đồng bằng sông Cửu Long (13)
    "Long An": ("Đồng bằng sông Cửu Long", "Tây Ninh"),
    "Tiền Giang": ("Đồng bằng sông Cửu Long", "Đồng Tháp"),
    "Bến Tre": ("Đồng bằng sông Cửu Long", "Vĩnh Long"),
    "Trà Vinh": ("Đồng bằng sông Cửu Long", "Vĩnh Long"),
    "Vĩnh Long": ("Đồng bằng sông Cửu Long", "Vĩnh Long"),
    "Đồng Tháp": ("Đồng bằng sông Cửu Long", "Đồng Tháp"),
    "An Giang": ("Đồng bằng sông Cửu Long", "An Giang"),
    "Kiên Giang": ("Đồng bằng sông Cửu Long", "An Giang"),
    "Cần Thơ": ("Đồng bằng sông Cửu Long", "Cần Thơ"),
    "Hậu Giang": ("Đồng bằng sông Cửu Long", "Cần Thơ"),
    "Sóc Trăng": ("Đồng bằng sông Cửu Long", "Cần Thơ"),
    "Bạc Liêu": ("Đồng bằng sông Cửu Long", "Cà Mau"),
    "Cà Mau": ("Đồng bằng sông Cửu Long", "Cà Mau"),
}

_COUNTRY_ALIASES: set[str] = {
    "cả nước",
    "toàn quốc",
    "tổng số",
    "tổng cộng",
    "việt nam",
    "whole country",
}

_REGION_ALIASES: dict[str, str] = {
    "đồng bằng sông hồng": "Đồng bằng sông Hồng",
    "đb sông hồng": "Đồng bằng sông Hồng",
    "đbsh": "Đồng bằng sông Hồng",
    "trung du và miền núi phía bắc": "Trung du và miền núi phía Bắc",
    "td và mn phía bắc": "Trung du và miền núi phía Bắc",
    "tdmnpb": "Trung du và miền núi phía Bắc",
    "bắc trung bộ và duyên hải miền trung": "Bắc Trung Bộ và Duyên hải miền Trung",
    "btb và dh miền trung": "Bắc Trung Bộ và Duyên hải miền Trung",
    "tây nguyên": "Tây Nguyên",
    "đông nam bộ": "Đông Nam Bộ",
    "đồng bằng sông cửu long": "Đồng bằng sông Cửu Long",
    "đb sông cửu long": "Đồng bằng sông Cửu Long",
    "đbscl": "Đồng bằng sông Cửu Long",
}

# Bilingual English suffixes printed alongside Vietnamese labels in GSO PDFs.
_BILINGUAL_SUFFIX_RE = re.compile(
    r"\s*[-–—]\s*("
    r"whole country|red river delta|northern midlands.*|"
    r"north central.*|central highlands|south east|mekong river delta"
    r")$",
    flags=re.IGNORECASE,
)

# Trailing footnote markers like (*), (1), (a).
_FOOTNOTE_RE = re.compile(r"\s*\([*0-9a-zA-Z]+\)\s*$")

# Explicit province aliases, abbreviations, historical names, and orthographic variants.
# Keys must be lowercased and normalized via normalize_unicode_vi.
_PROVINCE_ALIASES: dict[str, str] = {
    "tp.hcm": "TP. Hồ Chí Minh",
    "tphcm": "TP. Hồ Chí Minh",
    "tp hcm": "TP. Hồ Chí Minh",
    "tp. hcm": "TP. Hồ Chí Minh",
    "hồ chí minh": "TP. Hồ Chí Minh",
    "tp hồ chí minh": "TP. Hồ Chí Minh",
    "tp. hồ chí minh": "TP. Hồ Chí Minh",
    "thành phố hồ chí minh": "TP. Hồ Chí Minh",
    "hà nội": "Hà Nội",
    "tp hà nội": "Hà Nội",
    "tp. hà nội": "Hà Nội",
    "thành phố hà nội": "Hà Nội",
    "hà tây": "Hà Nội",
    "hà nội (cũ)": "Hà Nội",
    "thừa thiên huế": "Huế",
    "thừa thiên - huế": "Huế",
    "thừa thiên-huế": "Huế",
    "thành phố huế": "Huế",
    "tp huế": "Huế",
    "tp. huế": "Huế",
    "huế": "Huế",
    "bà rịa - vũng tàu": "Bà Rịa - Vũng Tàu",
    "bà rịa-vũng tàu": "Bà Rịa - Vũng Tàu",
    "bà rịa vũng tàu": "Bà Rịa - Vũng Tàu",
    "br-vt": "Bà Rịa - Vũng Tàu",
    "brvt": "Bà Rịa - Vũng Tàu",
    "đắc lắc": "Đắk Lắk",
    "đăk lăk": "Đắk Lắk",
    "đăk lắk": "Đắk Lắk",
    "đắc nông": "Đắk Nông",
    "đăk nông": "Đắk Nông",
    "bắc cạn": "Bắc Kạn",
    "hải phòng": "Hải Phòng",
    "tp hải phòng": "Hải Phòng",
    "tp. hải phòng": "Hải Phòng",
    "thành phố hải phòng": "Hải Phòng",
    "đà nẵng": "Đà Nẵng",
    "tp đà nẵng": "Đà Nẵng",
    "tp. đà nẵng": "Đà Nẵng",
    "thành phố đà nẵng": "Đà Nẵng",
    "cần thơ": "Cần Thơ",
    "tp cần thơ": "Cần Thơ",
    "tp. cần thơ": "Cần Thơ",
    "thành phố cần thơ": "Cần Thơ",
}

# Build lowercase canonical lookup for all 63 provinces.
_PROVINCE_LOWER_MAP: dict[str, str] = {
    name.lower(): name for name in PROVINCES_63
}
_PROVINCE_LOWER_MAP.update(_PROVINCE_ALIASES)

_PREFIX_STRIP_RE = re.compile(
    r"^(tỉnh|thành phố|tp\.?)\s+",
    flags=re.IGNORECASE,
)


def normalize_unicode_vi(text: str) -> str:
    """Normalize Vietnamese text: NFC, standard new-style tone marks, clean spaces."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    # Replace non-breaking spaces and collapse whitespace.
    s = s.replace("\u00a0", " ").replace("–", "-").replace("—", "-")
    # Keep only the first line if a bilingual PDF block merged English on line 2.
    if "\n" in s:
        s = s.splitlines()[0]
    s = re.sub(r"\s+", " ", s).strip()
    for old, new in _OLD_TO_NEW_TONE:
        if old in s:
            s = s.replace(old, new)
    s = _FOOTNOTE_RE.sub("", s).strip()
    return s


def classify_and_normalize_geo(name: str) -> tuple[str, GeoLevel]:
    """Classify a raw geographic label into (canonical_name, geo_level).

    Returns one of:
    - ("Cả nước", "country")
    - (<canonical region>, "region")
    - (<canonical 63-province name>, "province")
    - (<cleaned label>, "other")
    """
    cleaned = normalize_unicode_vi(name)
    if not cleaned:
        return ("", "other")

    cleaned_no_en = _BILINGUAL_SUFFIX_RE.sub("", cleaned).strip()
    key = cleaned_no_en.lower()

    if key in _COUNTRY_ALIASES:
        return ("Cả nước", "country")

    if key in _REGION_ALIASES:
        return (_REGION_ALIASES[key], "region")

    if key in _PROVINCE_LOWER_MAP:
        return (_PROVINCE_LOWER_MAP[key], "province")

    # Try stripping administrative prefix ("Tỉnh ", "Thành phố ", "TP. ")
    stripped = _PREFIX_STRIP_RE.sub("", cleaned_no_en).strip()
    stripped_key = stripped.lower()
    if stripped_key in _PROVINCE_LOWER_MAP:
        return (_PROVINCE_LOWER_MAP[stripped_key], "province")

    return (cleaned_no_en, "other")


def normalize_entity(name: str) -> str:
    """Return canonical entity name for evaluation and Self-Consistency clustering."""
    canonical, _ = classify_and_normalize_geo(name)
    return canonical


def normalize_entity_list(names: list[str]) -> list[str]:
    """Normalize an ordered list of entity names (e.g., top-k rankings)."""
    return [normalize_entity(n) for n in names]


def _default_audit_path() -> Path:
    """Locate docs/data_audit.json relative to the repository root."""
    return Path(__file__).resolve().parents[3] / "docs" / "data_audit.json"


def populate_admin_mapping(
    conn: sqlite3.Connection,
    audit_path: Path | None = None,
) -> int:
    """Populate `admin_mapping` idempotently from 63-province catalog + Phase 1 audit.

    Returns the total number of rows present in `admin_mapping` after population.
    """
    rows: list[tuple[str, str, str, str | None, str | None, str | None, str, str]] = []

    # 1. Country level
    for raw_country in ("Cả nước", "CẢ NƯỚC", "CẢ NƯỚC - WHOLE COUNTRY", "Toàn quốc"):
        rows.append((
            raw_country,
            "Cả nước",
            "country",
            None,
            "Cả nước",
            None,
            "canonical" if raw_country == "Cả nước" else "alias",
            "Cấp quốc gia (không đưa vào xếp hạng cấp tỉnh)",
        ))

    # 2. 6 socio-economic regions + bilingual PDF variants
    region_pdf_variants: dict[str, str] = {
        "Đồng bằng sông Hồng": "Đồng bằng sông Hồng - Red River Delta",
        "Trung du và miền núi phía Bắc": "Trung du và miền núi phía Bắc - Northern midlands and mountain areas",
        "Bắc Trung Bộ và Duyên hải miền Trung": "Bắc Trung Bộ và Duyên hải miền Trung - North Central and Central coastal areas",
        "Tây Nguyên": "Tây Nguyên - Central Highlands",
        "Đông Nam Bộ": "Đông Nam Bộ - South East",
        "Đồng bằng sông Cửu Long": "Đồng bằng sông Cửu Long - Mekong River Delta",
    }
    for region in REGIONS:
        rows.append((
            region,
            region,
            "region",
            region,
            region,
            None,
            "canonical",
            "Vùng kinh tế - xã hội (không đưa vào xếp hạng cấp tỉnh)",
        ))
        rows.append((
            region_pdf_variants[region],
            region,
            "region",
            region,
            region,
            None,
            "alias",
            "Nhãn song ngữ trong NG-TOAN-QUOC-2025-1.pdf",
        ))

    # 3. All 63 provinces (with region membership & 2025->2026 merger target)
    for province, (region, target_2026) in PROVINCES_63.items():
        if target_2026 != province:
            change_type = "merged_2025"
            effective_from = "2025-07-01"
            note = f"Sáp nhập vào {target_2026} theo NQ 202/2025/QH15 (Biểu 10/2026 dùng 34 tỉnh)"
        else:
            change_type = "canonical"
            effective_from = None
            note = "Đơn vị hành chính cấp tỉnh/thành phố trực thuộc Trung ương"
        rows.append((
            province,
            province,
            "province",
            region,
            target_2026,
            effective_from,
            change_type,
            note,
        ))

    # 4. Historical mergers, renames, and common orthographic/abbreviation variants
    special_variants: list[tuple[str, str, str, str | None, str, str]] = [
        ("Hà Tây", "Hà Nội", "Đồng bằng sông Hồng", "2008-08-01", "merged_historical", "Sáp nhập vào Hà Nội theo NQ 15/2008/QH12"),
        ("Thừa Thiên Huế", "Huế", "Bắc Trung Bộ và Duyên hải miền Trung", "2025-01-01", "renamed_2025", "Trở thành thành phố trực thuộc TW theo NQ 175/2024/QH15"),
        ("Thừa Thiên - Huế", "Huế", "Bắc Trung Bộ và Duyên hải miền Trung", "2025-01-01", "renamed_2025", "Biến thể gạch nối của Thừa Thiên Huế -> Huế"),
        ("Thành phố Huế", "Huế", "Bắc Trung Bộ và Duyên hải miền Trung", "2025-01-01", "alias", "Tên đầy đủ của Huế trong văn bản chính thức"),
        ("TP.HCM", "TP. Hồ Chí Minh", "Đông Nam Bộ", None, "alias", "Viết tắt phổ biến của TP. Hồ Chí Minh"),
        ("Thành phố Hồ Chí Minh", "TP. Hồ Chí Minh", "Đông Nam Bộ", None, "alias", "Tên đầy đủ của TP. Hồ Chí Minh"),
        ("Hồ Chí Minh", "TP. Hồ Chí Minh", "Đông Nam Bộ", None, "alias", "Tên rút gọn của TP. Hồ Chí Minh"),
        ("Bà Rịa-Vũng Tàu", "Bà Rịa - Vũng Tàu", "Đông Nam Bộ", None, "alias", "Biến thể không khoảng trắng quanh dấu gạch nối"),
        ("Hoà Bình", "Hòa Bình", "Trung du và miền núi phía Bắc", None, "alias", "Biến thể đặt dấu thanh kiểu cũ (oà -> òa)"),
        ("Thanh Hoá", "Thanh Hóa", "Bắc Trung Bộ và Duyên hải miền Trung", None, "alias", "Biến thể đặt dấu thanh kiểu cũ (oá -> óa)"),
        ("Khánh Hoà", "Khánh Hòa", "Bắc Trung Bộ và Duyên hải miền Trung", None, "alias", "Biến thể đặt dấu thanh kiểu cũ (oà -> òa)"),
        ("Đắc Lắc", "Đắk Lắk", "Tây Nguyên", None, "alias", "Biến thể chính tả cũ của Đắk Lắk"),
        ("Đắc Nông", "Đắk Nông", "Tây Nguyên", None, "alias", "Biến thể chính tả cũ của Đắk Nông"),
        ("Bắc Cạn", "Bắc Kạn", "Trung du và miền núi phía Bắc", None, "alias", "Biến thể chính tả cũ của Bắc Kạn"),
    ]
    for raw_name, canonical, region, eff, ctype, note in special_variants:
        target_2026 = PROVINCES_63[canonical][1]
        rows.append((raw_name, canonical, "province", region, target_2026, eff, ctype, note))

    # 5. Audit-driven verification: ingest every raw entity observed in docs/data_audit.json
    path = audit_path if audit_path is not None else _default_audit_path()
    if path.exists():
        audit_data = json.loads(path.read_text(encoding="utf-8"))
        excel_entities = audit_data.get("findings", {}).get("excel_2026_geo_entities", [])
        seen_raw = {r[0] for r in rows}
        for raw_ent in excel_entities:
            if raw_ent in seen_raw:
                continue
            canonical, geo_level = classify_and_normalize_geo(raw_ent)
            region = PROVINCES_63[canonical][0] if canonical in PROVINCES_63 else None
            target_2026 = PROVINCES_63[canonical][1] if canonical in PROVINCES_63 else canonical
            rows.append((
                raw_ent,
                canonical,
                geo_level,
                region,
                target_2026,
                "2025-07-01" if geo_level == "province" else None,
                "audit_excel_2026",
                "Thực thể trích từ 2.-Bieu-thang-10.2026.xlsx qua Phase 1 Data Audit",
            ))
            seen_raw.add(raw_ent)

    conn.executemany(
        """
        INSERT OR REPLACE INTO admin_mapping
            (raw_name, canonical_name, geo_level, region_name, canonical_2026,
             effective_from, change_type, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM admin_mapping").fetchone()[0]
    return int(count)

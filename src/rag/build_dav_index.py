import json
import re
from collections import defaultdict
from pathlib import Path
import openpyxl


DATA_DIR = Path(__file__).resolve().parent / "data"
XLSX_PATH = DATA_DIR / "dav_55k.xlsx"
OUT_JSONL = DATA_DIR / "dav_55k_brands.jsonl"


def _clean(val) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def build_dav_jsonl(xlsx_path: Path = XLSX_PATH, out_path: Path = OUT_JSONL) -> int:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active

    grouped: dict[str, dict] = {}
    total_rows = 0

    for idx, row in enumerate(ws.iter_rows(values_only=True)):
        if idx == 0:
            continue
        total_rows += 1
        gplh = _clean(row[1]) if len(row) > 1 else ""
        status = _clean(row[2]) if len(row) > 2 else ""
        gplh_old = _clean(row[3]) if len(row) > 3 else ""
        brand = _clean(row[4]) if len(row) > 4 else ""
        rx_type = _clean(row[5]) if len(row) > 5 else ""
        ingredient = _clean(row[7]) if len(row) > 7 else ""
        strength = _clean(row[8]) if len(row) > 8 else ""
        dosage_form = _clean(row[9]) if len(row) > 9 else ""
        manufacturer = _clean(row[24]) if len(row) > 24 else ""
        country = _clean(row[25]) if len(row) > 25 else ""
        detail_url = _clean(row[30]) if len(row) > 30 else ""

        if not brand or len(brand) < 2:
            continue

        key = brand.lower()
        if key not in grouped:
            grouped[key] = {
                "brand_name": brand,
                "active_ingredient": ingredient,
                "rx_type": rx_type,
                "status": status,
                "strengths": [],
                "dosage_forms": [],
                "gplh_all": [],
                "manufacturer": manufacturer,
                "country": country,
                "detail_url": detail_url,
            }

        entry = grouped[key]
        if ingredient and not entry["active_ingredient"]:
            entry["active_ingredient"] = ingredient
        if strength and strength not in entry["strengths"] and len(entry["strengths"]) < 5:
            entry["strengths"].append(strength)
        if dosage_form and dosage_form not in entry["dosage_forms"] and len(entry["dosage_forms"]) < 3:
            entry["dosage_forms"].append(dosage_form)
        if gplh and gplh not in entry["gplh_all"]:
            entry["gplh_all"].append(gplh)
        if gplh_old and gplh_old not in entry["gplh_all"]:
            entry["gplh_all"].append(gplh_old)

    wb.close()

    with out_path.open("w", encoding="utf-8") as f:
        for entry in grouped.values():
            strengths_str = "; ".join(entry["strengths"]) or "Xem trên bao bì"
            forms_str = "; ".join(entry["dosage_forms"]) or "N/A"
            gplh_str = ", ".join(entry["gplh_all"][:5]) or "N/A"
            mfr_str = f"{entry['manufacturer']} ({entry['country']})" if entry["country"] else entry["manufacturer"]

            rag_text = (
                f"BIỆT DƯỢC ĐĂNG KÝ BỘ Y TẾ (DAV): {entry['brand_name']}\n"
                f"- Hoạt chất chính: {entry['active_ingredient'] or 'Chưa ghi rõ'}\n"
                f"- Phân loại kê đơn: {entry['rx_type'] or 'N/A'} | Trạng thái: {entry['status'] or 'Hiệu lực'}\n"
                f"- Hàm lượng công bố: {strengths_str}\n"
                f"- Dạng bào chế: {forms_str}\n"
                f"- Số GPLH (DAV): {gplh_str}\n"
                f"- Cơ sở sản xuất: {mfr_str or 'N/A'}"
            )
            if entry["detail_url"]:
                rag_text += f"\n- Tra cứu DAV: {entry['detail_url']}"

            record = {
                "brand_name": entry["brand_name"],
                "active_ingredient": entry["active_ingredient"],
                "rx_type": entry["rx_type"],
                "gplh_all": entry["gplh_all"],
                "rag_text": rag_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {total_rows} DAV rows -> {len(grouped)} unique brands at {out_path}")
    return len(grouped)


if __name__ == "__main__":
    build_dav_jsonl()

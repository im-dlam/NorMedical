import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import build_application
from src.domain.models import ChatMessage


def run_edge_case_tests():
    svc, _ = build_application()
    cases = [
        ("thuốc nào cho ho", "TRA CỨU DƯỢC LÝ", True),
        ("đau đầu uống thuốc gì", "TRA CỨU DƯỢC LÝ", True),
        ("tiêu chảy uống thuốc gì", "TRA CỨU DƯỢC LÝ", True),
        ("liều dùng paracetamol", "TRA CỨU DƯỢC LÝ", True),
        ("omeprazol uống lúc nào", "TRA CỨU DƯỢC LÝ", True),
        ("thuốc trị dị ứng mề đay", "TRA CỨU DƯỢC LÝ", True),
        ("táo bón nên uống thuốc gì", "TRA CỨU DƯỢC LÝ", True),
        ("tôi bị đau thắt ngực khó thở dữ dội", "CỜ ĐỎ CẤP CỨU", False),
        ("hôm qua đi ngủ tôi có uống nước đa", "LƯỢT 1", False),
        ("sáng dậy tôi bị đau học", "LƯỢT 1", False),
    ]

    passed = 0
    for query, expected_mode, expect_drugs in cases:
        svc.reset_session()
        norm = svc._normalizer.normalize(query)
        docs = svc._select_rag_docs(norm)
        directive = svc._build_phase_directive(norm, docs)
        drugs = [d.title for d in docs if d.doc_type == "drug"]

        ok_mode = expected_mode in directive
        ok_drugs = (len(drugs) > 0) if expect_drugs else (len(drugs) == 0)
        status = "PASS" if (ok_mode and ok_drugs) else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] '{query}' -> norm='{norm}' | mode_ok={ok_mode} | drugs={drugs}")

    # Test multi-turn anti-loop (Case 11 & 12)
    svc.reset_session()
    svc._history.append(ChatMessage("user", "thấy to béo mặt"))
    svc._history.append(ChatMessage("assistant", "Bạn có kèm sưng đau không?"))
    d_neg = svc._build_phase_directive("không á", [])
    d_info = svc._build_phase_directive("bị khoảng 2 tháng nay rồi", [])
    ok_11 = "KẾT LUẬN" in d_neg
    ok_12 = "LƯỢT 2" in d_info
    passed += int(ok_11) + int(ok_12)
    print(f"[{'PASS' if ok_11 else 'FAIL'}] Multi-turn short negative ('không á') -> KẾT LUẬN")
    print(f"[{'PASS' if ok_12 else 'FAIL'}] Multi-turn new info ('bị khoảng 2 tháng...') -> LƯỢT 2")

    total = len(cases) + 2
    print(f"\n=== TỔNG KẾT TEST: {passed}/{total} PASSED ===")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    run_edge_case_tests()

from src.services.consultation_service import ConsultationService


def run_cli(service: ConsultationService, api_url: str) -> None:
    print(f"Loaded {service.knowledge_count} tài liệu Dược thư Quốc gia & Bệnh học Vinmec.")
    print(f" Kết nối LM Studio API: {api_url}")
    print("Gõ 'reset' để khám ca mới, gõ 'exit' để thoát.")
    print("=" * 85)

    while True:
        try:
            raw_in = input("\n BỆNH NHÂN: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[L] Đã kết thúc phiên khám.")
            break

        if not raw_in or raw_in.lower() in {"exit", "quit", "thoat", "q"}:
            print("[L] Đã kết thúc phiên khám.")
            break

        if raw_in.lower() in {"reset", "moi", "clear"}:
            service.reset_session()
            print("[L] Đã làm mới bệnh án!")
            continue

        docs, token_stream = service.stream_turn(raw_in)
        if docs:
            summary = ", ".join(f"{d.doc_type.upper()}: {d.title}" for d in docs)
            print(f"[AI RAG TRA CỨU]: {summary}")

        print("\nRAUMA MEDICAL:")
        try:
            for token in token_stream:
                print(token, end="", flush=True)
            print()
        except ConnectionError as err:
            print(f" {err}")
        print("-" * 85)

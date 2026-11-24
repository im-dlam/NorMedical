import argparse
from pathlib import Path
from src.config.settings import load_settings
from src.llm.lmstudio_client import LMStudioSSEClient
from src.preprocessing.normalizer import VietnameseMedicalNormalizer
from src.rag.bm25_retriever import BM25MedicalRetriever
from src.services.consultation_service import ConsultationService
from src.ui.cli import run_cli


def build_application() -> tuple[ConsultationService, str]:
    """Composition Root: Lắp ráp toàn bộ dependencies theo nguyên lý Dependency Injection."""
    settings = load_settings(Path(__file__).resolve().parent)
    normalizer = VietnameseMedicalNormalizer(settings.nlp)
    retriever = BM25MedicalRetriever(settings.rag, settings.nlp)
    llm_client = LMStudioSSEClient(settings.api, settings.decoding, settings.keys)
    service = ConsultationService(
        settings=settings,
        normalizer=normalizer,
        retriever=retriever,
        llm_client=llm_client,
    )
    return service, settings.api.base_url


def main() -> None:
    parser = argparse.ArgumentParser(description="RaumaMedical q3AI Runner")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", help="Chế độ chạy: cli hoặc web (Gradio)")
    args = parser.parse_args()

    service, api_url = build_application()
    if args.mode == "web":
        from src.ui.web_gradio import run_web_ui
        run_web_ui(service)
    else:
        run_cli(service, api_url)


if __name__ == "__main__":
    main()

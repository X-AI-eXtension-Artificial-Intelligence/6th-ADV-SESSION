def setup_all():
    """
    실행 순서:
    1) 모델 다운로드
    2) CSV 청킹
    3) FAISS DB 구축
    """
    from models.setup_model import download_model_files
    from data.raw.chunking import main as chunk_main
    from data.vector_db.setup_db import main as db_main

    print("[1/3] 모델 다운로드 시작...")
    download_model_files()
    print("[1/3] 모델 다운로드 완료.")

    print("[2/3] 청킹 시작...")
    chunk_main()
    print("[2/3] 청킹 완료.")

    print("[3/3] 벡터 DB 구축 시작...")
    db_main()
    print("[3/3] 벡터 DB 구축 완료.")

from setuptools import setup, find_packages

setup(
    name="legal-chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",                 # CPU/GPU 둘 다 가능
        "huggingface-hub",
        "sentence-transformers",
        "python-dotenv",
        "pandas",
        "numpy",
        "faiss-cpu",             # 기본 CPU 버전
        "tqdm",
        "gdown"
    ],
    entry_points={
        "console_scripts": [
            "setup-models=models.setup_model:download_model_files",
            "chunk-laws=data.raw.chunking:main",
            "setup-db=data.vector_db.setup_db:main",
            "setup-all=legal_chatbot.cli:setup_all",
        ]
    }
)

from setuptools import setup, find_packages

setup(
    name="legal-chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "huggingface-hub"
    ],
    entry_points={
        "console_scripts": [
            "download-models=models.setup_model:download_model_files"
        ]
    }
)

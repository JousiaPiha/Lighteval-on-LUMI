# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

setup(
    name="lighteval",
    version="0.6.0.dev0",
    packages=find_packages(where="src"),  # Ensures packages in "src" are found
    package_dir={"": "src"},  # Maps the root of the package to "src"
    entry_points={
        "console_scripts": [
            "lighteval=lighteval.__main__:cli_evaluate",  # Adds the CLI script
        ],
    },
    extras_require={
        "accelerate": ["accelerate"],
        "vllm": ["vllm", "ray", "more_itertools"],
    },
    install_requires=[
        # List all base dependencies here
        "transformers>=4.38.0",
        "huggingface_hub>=0.23.0",
        "torch>=2.0,<2.5",
        "GitPython>=3.1.41",
        "datasets>=2.14.0",
        "termcolor==2.3.0",
        "pytablewriter",
        "colorama",
        "aenum==3.1.15",
        "nltk==3.9.1",
        "scikit-learn",
        "spacy==3.7.2",
        "sacrebleu",
        "rouge_score==0.1.2",
        "sentencepiece>=0.1.99",
        "protobuf==3.20.*",
        "pycountry",
        "fsspec>=2023.12.2",
    ],
)

from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass, field
from pathlib import Path


class PythonSourceEncodingError(ValueError):
    """Python kaynağının PEP 263 encoding bilgisi güvenle çözümlenemedi."""


@dataclass(frozen=True, slots=True)
class DecodedPythonSource:
    text: str = field(repr=False)
    encoding: str


def read_python_source(file_path: str | Path) -> DecodedPythonSource:
    """Bir Python dosyasını tokenizer ile aynı PEP 263 politikasında okur."""
    path = Path(file_path)
    try:
        with tokenize.open(path) as stream:
            return DecodedPythonSource(
                text=stream.read(),
                encoding=stream.encoding,
            )
    except (LookupError, UnicodeError, SyntaxError) as error:
        raise PythonSourceEncodingError(
            "Python source encoding çözümlenemedi."
        ) from error


def decode_python_source_bytes(file_bytes: bytes) -> DecodedPythonSource:
    """Python kaynak byte'larını PEP 263 kurallarıyla decode eder."""
    if not isinstance(file_bytes, bytes):
        raise TypeError("file_bytes bytes olmalıdır.")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(file_bytes).readline)
        return DecodedPythonSource(
            text=file_bytes.decode(encoding),
            encoding=encoding,
        )
    except (LookupError, UnicodeError, SyntaxError) as error:
        raise PythonSourceEncodingError(
            "Python source encoding çözümlenemedi."
        ) from error

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path


class GeneratedTestFileWriter:
    """Üretilen pytest kaynak kodunu güvenli biçimde dosyaya yazar."""

    def write(
        self,
        source_code: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Python kaynak kodunu belirtilen dosyaya kaydeder.

        Yazma işleminden önce kaynak kodun geçerli Python sözdizimine
        sahip olduğu doğrulanır. Dosya, geçici bir dosya üzerinden atomik
        olarak oluşturulur.

        Args:
            source_code: Dosyaya yazılacak Python kaynak kodu.
            output_path: Oluşturulacak ``.py`` dosyasının yolu.
            overwrite: Mevcut dosyanın üzerine yazılmasına izin verilip
                verilmediği.

        Returns:
            Oluşturulan dosyanın Path nesnesi.

        Raises:
            TypeError: Kaynak kod veya çıktı yolu geçersiz türdeyse.
            ValueError: Kaynak kod boşsa, Python sözdizimi geçersizse
                veya çıktı uzantısı ``.py`` değilse.
            FileExistsError: Dosya mevcutken overwrite False ise.
        """
        normalized_source = self._validate_source_code(source_code)
        path = self._normalize_output_path(output_path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Çıktı dosyası zaten mevcut: {path}"
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_atomically(
            source_code=normalized_source,
            output_path=path,
        )

        return path

    @staticmethod
    def _validate_source_code(source_code: str) -> str:
        """
        Python kaynak kodunu doğrular.

        Args:
            source_code: Doğrulanacak kaynak kod.

        Returns:
            Doğrulanmış kaynak kod.

        Raises:
            TypeError: Kaynak kod string değilse.
            ValueError: Kaynak kod boşsa veya sözdizimi geçersizse.
        """
        if not isinstance(source_code, str):
            raise TypeError(
                "Kaynak kod string türünde olmalıdır."
            )

        if not source_code.strip():
            raise ValueError("Kaynak kod boş olamaz.")

        try:
            ast.parse(source_code)
        except SyntaxError as error:
            raise ValueError(
                "Kaynak kod geçerli Python sözdizimine sahip değil."
            ) from error

        return source_code

    @staticmethod
    def _normalize_output_path(
        output_path: str | Path,
    ) -> Path:
        """
        Çıktı dosyası yolunu doğrular ve Path nesnesine dönüştürür.

        Args:
            output_path: Doğrulanacak çıktı yolu.

        Returns:
            Normalize edilmiş Path nesnesi.

        Raises:
            TypeError: Yol string veya Path değilse.
            ValueError: Yol boşsa veya uzantısı ``.py`` değilse.
        """
        if not isinstance(output_path, (str, Path)):
            raise TypeError(
                "Çıktı yolu string veya Path türünde olmalıdır."
            )

        if isinstance(output_path, str) and not output_path.strip():
            raise ValueError("Çıktı yolu boş olamaz.")

        path = Path(output_path)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Çıktı dosyasının uzantısı .py olmalıdır."
            )

        return path

    @staticmethod
    def _write_atomically(
        source_code: str,
        output_path: Path,
    ) -> None:
        """
        Kaynak kodu geçici dosya üzerinden atomik biçimde yazar.

        Args:
            source_code: Yazılacak Python kaynak kodu.
            output_path: Nihai çıktı dosyasının yolu.
        """
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=f".{output_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(source_code)
                temporary_path = Path(temporary_file.name)

            os.replace(
                temporary_path,
                output_path,
            )
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

            raise
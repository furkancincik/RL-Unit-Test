from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path


class GeneratedTestFileWriter:
    """Ãœretilen pytest kaynak kodunu gÃ¼venli biÃ§imde dosyaya yazar."""

    def write(
        self,
        source_code: str,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Python kaynak kodunu belirtilen dosyaya kaydeder.

        Yazma iÅŸleminden Ã¶nce kaynak kodun geÃ§erli Python sÃ¶zdizimine
        sahip olduÄŸu doÄŸrulanÄ±r. Dosya, geÃ§ici bir dosya Ã¼zerinden atomik
        olarak oluÅŸturulur.

        Args:
            source_code: Dosyaya yazÄ±lacak Python kaynak kodu.
            output_path: OluÅŸturulacak ``.py`` dosyasÄ±nÄ±n yolu.
            overwrite: Mevcut dosyanÄ±n Ã¼zerine yazÄ±lmasÄ±na izin verilip
                verilmediÄŸi.

        Returns:
            OluÅŸturulan dosyanÄ±n Path nesnesi.

        Raises:
            TypeError: Kaynak kod veya Ã§Ä±ktÄ± yolu geÃ§ersiz tÃ¼rdeyse.
            ValueError: Kaynak kod boÅŸsa, Python sÃ¶zdizimi geÃ§ersizse
                veya Ã§Ä±ktÄ± uzantÄ±sÄ± ``.py`` deÄŸilse.
            FileExistsError: Dosya mevcutken overwrite False ise.
        """
        normalized_source = self._validate_source_code(source_code)
        path = self._normalize_output_path(output_path)

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Ã‡Ä±ktÄ± dosyasÄ± zaten mevcut: {path}"
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
        Python kaynak kodunu doÄŸrular.

        Args:
            source_code: DoÄŸrulanacak kaynak kod.

        Returns:
            DoÄŸrulanmÄ±ÅŸ kaynak kod.

        Raises:
            TypeError: Kaynak kod string deÄŸilse.
            ValueError: Kaynak kod boÅŸsa veya sÃ¶zdizimi geÃ§ersizse.
        """
        if not isinstance(source_code, str):
            raise TypeError(
                "Kaynak kod string tÃ¼rÃ¼nde olmalÄ±dÄ±r."
            )

        if not source_code.strip():
            raise ValueError("Kaynak kod boÅŸ olamaz.")

        try:
            ast.parse(source_code)
        except SyntaxError as error:
            raise ValueError(
                "Kaynak kod geÃ§erli Python sÃ¶zdizimine sahip deÄŸil."
            ) from error

        return source_code

    @staticmethod
    def _normalize_output_path(
        output_path: str | Path,
    ) -> Path:
        """
        Ã‡Ä±ktÄ± dosyasÄ± yolunu doÄŸrular ve Path nesnesine dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r.

        Args:
            output_path: DoÄŸrulanacak Ã§Ä±ktÄ± yolu.

        Returns:
            Normalize edilmiÅŸ Path nesnesi.

        Raises:
            TypeError: Yol string veya Path deÄŸilse.
            ValueError: Yol boÅŸsa veya uzantÄ±sÄ± ``.py`` deÄŸilse.
        """
        if not isinstance(output_path, (str, Path)):
            raise TypeError(
                "Ã‡Ä±ktÄ± yolu string veya Path tÃ¼rÃ¼nde olmalÄ±dÄ±r."
            )

        if isinstance(output_path, str) and not output_path.strip():
            raise ValueError("Ã‡Ä±ktÄ± yolu boÅŸ olamaz.")

        path = Path(output_path)

        if path.suffix.lower() != ".py":
            raise ValueError(
                "Ã‡Ä±ktÄ± dosyasÄ±nÄ±n uzantÄ±sÄ± .py olmalÄ±dÄ±r."
            )

        return path

    @staticmethod
    def _write_atomically(
        source_code: str,
        output_path: Path,
    ) -> None:
        """
        Kaynak kodu geÃ§ici dosya Ã¼zerinden atomik biÃ§imde yazar.

        Args:
            source_code: YazÄ±lacak Python kaynak kodu.
            output_path: Nihai Ã§Ä±ktÄ± dosyasÄ±nÄ±n yolu.
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


from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from cfg.path_analyzer import ExecutionPath


@dataclass(frozen=True, slots=True)
class SymbolicVariableState:
    variable_name: str
    exact_value: float | None
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True, slots=True)
class PathSymbolicState:
    variables: tuple[SymbolicVariableState, ...]

    def get_variable(
        self,
        variable_name: str,
    ) -> SymbolicVariableState | None:
        for variable in self.variables:
            if variable.variable_name == variable_name:
                return variable

        return None


@dataclass(slots=True)
class _MutableVariableState:
    exact_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None


class PathStateAnalyzer:
    """
    Tek bir ExecutionPath boyunca statik olarak izlenebilen
    basit sayısal değişken state'lerini çıkarır.

    v4.4 ile desteklenen başlıca yapılar:
    - x = literal
    - x += literal
    - x -= literal
    - Path üzerinde görülen branch-local assignment/update takibi
    - Loop içindeki augmented assignment'lar için conservative widening

    Bu sınıf tam symbolic execution yapmaz.
    Özellikle loop iteration sayısı bilinmiyorsa kesin değer uydurmaz;
    yalnızca güvenli biçimde çıkarılabilen path-local exact/range state'i
    üretir.
    """

    def analyze_file(
        self,
        *,
        source_file: Path,
        function_name: str,
        path: ExecutionPath,
    ) -> PathSymbolicState:
        if not isinstance(source_file, Path):
            raise TypeError(
                "source_file Path olmalıdır."
            )

        if not source_file.exists():
            raise FileNotFoundError(
                source_file
            )

        source = source_file.read_text(
            encoding="utf-8"
        )

        return self.analyze_source(
            source=source,
            function_name=function_name,
            path=path,
        )

    def analyze_source(
        self,
        *,
        source: str,
        function_name: str,
        path: ExecutionPath,
    ) -> PathSymbolicState:
        if not isinstance(source, str):
            raise TypeError(
                "source string olmalıdır."
            )

        if not isinstance(function_name, str):
            raise TypeError(
                "function_name string olmalıdır."
            )

        if not function_name.strip():
            raise ValueError(
                "function_name boş olamaz."
            )

        if not isinstance(
            path,
            ExecutionPath,
        ):
            raise TypeError(
                "path bir ExecutionPath örneği olmalıdır."
            )

        tree = ast.parse(
            source
        )

        function_node = self._find_function(
            tree=tree,
            function_name=function_name,
        )

        path_line_numbers = {
            line_number
            for line_number in path.line_numbers
            if line_number is not None
        }

        mutable_states: dict[
            str,
            _MutableVariableState,
        ] = {}

        self._walk_statements(
            statements=function_node.body,
            path_line_numbers=path_line_numbers,
            states=mutable_states,
            inside_loop=False,
        )

        variables = tuple(
            SymbolicVariableState(
                variable_name=variable_name,
                exact_value=state.exact_value,
                lower_bound=state.lower_bound,
                upper_bound=state.upper_bound,
            )
            for variable_name, state
            in sorted(
                mutable_states.items()
            )
        )

        return PathSymbolicState(
            variables=variables
        )

    @staticmethod
    def _find_function(
        *,
        tree: ast.Module,
        function_name: str,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        for node in tree.body:
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                if node.name == function_name:
                    return node

        raise ValueError(
            f"Fonksiyon bulunamadı: {function_name}"
        )

    def _walk_statements(
        self,
        *,
        statements: list[ast.stmt],
        path_line_numbers: set[int],
        states: dict[
            str,
            _MutableVariableState,
        ],
        inside_loop: bool,
    ) -> None:
        for statement in statements:

            line_number = getattr(
                statement,
                "lineno",
                None,
            )

            if isinstance(
                statement,
                ast.Assign,
            ):
                if (
                    line_number
                    in path_line_numbers
                ):
                    self._handle_assign(
                        statement=statement,
                        states=states,
                    )

            elif isinstance(
                statement,
                ast.AnnAssign,
            ):
                if (
                    line_number
                    in path_line_numbers
                ):
                    self._handle_ann_assign(
                        statement=statement,
                        states=states,
                    )

            elif isinstance(
                statement,
                ast.AugAssign,
            ):
                if (
                    line_number
                    in path_line_numbers
                ):
                    self._handle_aug_assign(
                        statement=statement,
                        states=states,
                        inside_loop=inside_loop,
                    )

            elif isinstance(
                statement,
                ast.If,
            ):
                self._walk_statements(
                    statements=statement.body,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

                self._walk_statements(
                    statements=statement.orelse,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                ),
            ):
                self._walk_statements(
                    statements=statement.body,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=True,
                )

                self._walk_statements(
                    statements=statement.orelse,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

            elif isinstance(
                statement,
                ast.Try,
            ):
                self._walk_statements(
                    statements=statement.body,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

                for handler in (
                    statement.handlers
                ):
                    self._walk_statements(
                        statements=handler.body,
                        path_line_numbers=(
                            path_line_numbers
                        ),
                        states=states,
                        inside_loop=inside_loop,
                    )

                self._walk_statements(
                    statements=statement.orelse,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

                self._walk_statements(
                    statements=statement.finalbody,
                    path_line_numbers=(
                        path_line_numbers
                    ),
                    states=states,
                    inside_loop=inside_loop,
                )

    def _handle_assign(
        self,
        *,
        statement: ast.Assign,
        states: dict[
            str,
            _MutableVariableState,
        ],
    ) -> None:
        value = self._extract_numeric_literal(
            statement.value
        )

        for target in statement.targets:
            if not isinstance(
                target,
                ast.Name,
            ):
                continue

            if value is None:
                states.pop(
                    target.id,
                    None,
                )
                continue

            states[
                target.id
            ] = _MutableVariableState(
                exact_value=value,
                lower_bound=value,
                upper_bound=value,
            )

    def _handle_ann_assign(
        self,
        *,
        statement: ast.AnnAssign,
        states: dict[
            str,
            _MutableVariableState,
        ],
    ) -> None:
        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return

        if statement.value is None:
            return

        value = self._extract_numeric_literal(
            statement.value
        )

        if value is None:
            states.pop(
                statement.target.id,
                None,
            )
            return

        states[
            statement.target.id
        ] = _MutableVariableState(
            exact_value=value,
            lower_bound=value,
            upper_bound=value,
        )

    def _handle_aug_assign(
        self,
        *,
        statement: ast.AugAssign,
        states: dict[
            str,
            _MutableVariableState,
        ],
        inside_loop: bool,
    ) -> None:
        if not isinstance(
            statement.target,
            ast.Name,
        ):
            return

        variable_name = (
            statement.target.id
        )

        state = states.get(
            variable_name
        )

        if state is None:
            return

        delta = self._extract_numeric_literal(
            statement.value
        )

        if delta is None:
            states.pop(
                variable_name,
                None,
            )
            return

        if isinstance(
            statement.op,
            ast.Add,
        ):
            effective_delta = delta

        elif isinstance(
            statement.op,
            ast.Sub,
        ):
            effective_delta = -delta

        else:
            states.pop(
                variable_name,
                None,
            )
            return

        if inside_loop:
            self._apply_loop_delta_conservatively(
                state=state,
                delta=effective_delta,
            )
            return

        self._apply_delta(
            state=state,
            delta=effective_delta,
        )

    @staticmethod
    def _apply_loop_delta_conservatively(
        *,
        state: _MutableVariableState,
        delta: float,
    ) -> None:
        """
        Loop içindeki ``x += c`` / ``x -= c`` güncellemesini, iteration
        sayısını bilmeden güvenli biçimde uygular.

        Path update satırını içerdiği için en az bir güncelleme gerçekleşmiş
        kabul edilir; ancak kaç kez tekrarlandığı bilinmediğinden exact value
        korunmaz ve açık uçlu bir bound üretilir.

        Örnek:
            x = 0
            loop içinde x += 1

        Sonuç:
            exact=None
            lower_bound=1
            upper_bound=None
        """
        if delta == 0:
            # Sıfır delta kaç kez uygulanırsa uygulansın state değişmez.
            return

        if delta > 0:
            base_lower = (
                state.exact_value
                if state.exact_value is not None
                else state.lower_bound
            )

            state.exact_value = None

            if base_lower is not None:
                state.lower_bound = (
                    base_lower + delta
                )
            else:
                state.lower_bound = None

            # Kaç iterasyon olduğu bilinmediği için güvenli üst sınır yok.
            state.upper_bound = None
            return

        base_upper = (
            state.exact_value
            if state.exact_value is not None
            else state.upper_bound
        )

        state.exact_value = None

        if base_upper is not None:
            state.upper_bound = (
                base_upper + delta
            )
        else:
            state.upper_bound = None

        # Negatif yönde kaç iterasyon olduğu bilinmediği için alt sınır yok.
        state.lower_bound = None

    @staticmethod
    def _apply_delta(
        *,
        state: _MutableVariableState,
        delta: float,
    ) -> None:
        if state.exact_value is not None:
            state.exact_value += delta

        if state.lower_bound is not None:
            state.lower_bound += delta

        if state.upper_bound is not None:
            state.upper_bound += delta

    @staticmethod
    def _extract_numeric_literal(
        node: ast.AST,
    ) -> float | None:
        if (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                (int, float),
            )
            and not isinstance(
                node.value,
                bool,
            )
        ):
            return float(
                node.value
            )

        if (
            isinstance(
                node,
                ast.UnaryOp,
            )
            and isinstance(
                node.op,
                (
                    ast.USub,
                    ast.UAdd,
                ),
            )
            and isinstance(
                node.operand,
                ast.Constant,
            )
            and isinstance(
                node.operand.value,
                (int, float),
            )
            and not isinstance(
                node.operand.value,
                bool,
            )
        ):
            value = float(
                node.operand.value
            )

            if isinstance(
                node.op,
                ast.USub,
            ):
                return -value

            return value

        return None

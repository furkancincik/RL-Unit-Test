from __future__ import annotations

from cfg.path_analyzer import ExecutionPath
from cfg.path_feasibility_analyzer import (
    FeasibilityStatus,
    PathFeasibilityResult,
)
from models.coverage_reachability_result import (
    FunctionCoverageReachabilityResult,
    LineReachabilityEvidence,
    LineReachabilityStatus,
)
from models.coverage_result import FunctionCoverageResult


class CoverageReachabilityService:
    """
    Fonksiyon satırlarını coverage ve bounded path analiziyle sınıflandırır.

    Servis branch reachability kararı üretmez. Coverage.py branch uçları ile
    CFG kenarları her Python yapısında bire bir eşleşmediğinden, branch
    ayrıntıları mevcut aşamada ham coverage verisi olarak korunur.
    """

    def analyze(
        self,
        *,
        coverage_result: FunctionCoverageResult,
        paths: tuple[ExecutionPath, ...],
        feasibility_results: tuple[PathFeasibilityResult, ...],
        max_visits_per_node: int,
    ) -> FunctionCoverageReachabilityResult:
        """
        Coverage satırlarını path feasibility kanıtlarıyla eşleştirir.

        Sınıflandırma önceliği şöyledir:

        1. Gerçek çalıştırmada kapsanan satır ``COVERED`` olur.
        2. En az bir FEASIBLE path içeren eksik satır
           ``FEASIBLE_UNCOVERED`` olur.
        3. UNKNOWN path, eksik metadata veya path eşleşmesinin bulunmaması
           ``UNRESOLVED`` sonucunu üretir.
        4. Yalnızca metadata'sı tam INFEASIBLE path'lerde görülen satır
           ``INFEASIBLE_ONLY`` olur.
        """
        self._validate_coverage_result(
            coverage_result
        )
        self._validate_paths(paths)
        self._validate_feasibility_results(
            feasibility_results
        )
        self._validate_path_result_count(
            paths=paths,
            feasibility_results=feasibility_results,
        )
        self._validate_max_visits(
            max_visits_per_node
        )

        path_line_sets = tuple(
            self._extract_path_lines(path)
            for path in paths
        )
        path_metadata_complete = all(
            path_lines is not None
            for path_lines in path_line_sets
        )

        covered_line_set = set(
            coverage_result.covered_lines
        )
        executable_lines = tuple(
            sorted(
                covered_line_set
                | set(coverage_result.missing_lines)
            )
        )

        line_evidence = tuple(
            self._classify_line(
                line_number=line_number,
                is_covered=(
                    line_number in covered_line_set
                ),
                path_line_sets=path_line_sets,
                feasibility_results=(
                    feasibility_results
                ),
                path_metadata_complete=(
                    path_metadata_complete
                ),
            )
            for line_number in executable_lines
        )

        return FunctionCoverageReachabilityResult(
            coverage_result=coverage_result,
            line_evidence=line_evidence,
            analyzed_path_count=len(paths),
            max_visits_per_node=max_visits_per_node,
            path_metadata_complete=path_metadata_complete,
        )

    @classmethod
    def _classify_line(
        cls,
        *,
        line_number: int,
        is_covered: bool,
        path_line_sets: tuple[frozenset[int] | None, ...],
        feasibility_results: tuple[PathFeasibilityResult, ...],
        path_metadata_complete: bool,
    ) -> LineReachabilityEvidence:
        """Tek bir coverage satırı için durum ve path kanıtı üretir."""
        (
            feasible_indices,
            infeasible_indices,
            unknown_indices,
        ) = cls._collect_path_indices(
            line_number=line_number,
            path_line_sets=path_line_sets,
            feasibility_results=feasibility_results,
        )

        if is_covered:
            status = LineReachabilityStatus.COVERED
        elif feasible_indices:
            status = (
                LineReachabilityStatus.FEASIBLE_UNCOVERED
            )
        elif (
            unknown_indices
            or not path_metadata_complete
            or not infeasible_indices
        ):
            status = LineReachabilityStatus.UNRESOLVED
        else:
            status = LineReachabilityStatus.INFEASIBLE_ONLY

        return LineReachabilityEvidence(
            line_number=line_number,
            status=status,
            feasible_path_indices=feasible_indices,
            infeasible_path_indices=infeasible_indices,
            unknown_path_indices=unknown_indices,
        )

    @staticmethod
    def _collect_path_indices(
        *,
        line_number: int,
        path_line_sets: tuple[frozenset[int] | None, ...],
        feasibility_results: tuple[PathFeasibilityResult, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Satırı içeren path indekslerini feasibility durumuna ayırır."""
        feasible_indices: list[int] = []
        infeasible_indices: list[int] = []
        unknown_indices: list[int] = []

        for path_index, (
            path_lines,
            feasibility_result,
        ) in enumerate(
            zip(
                path_line_sets,
                feasibility_results,
                strict=True,
            ),
            start=1,
        ):
            if (
                path_lines is None
                or line_number not in path_lines
            ):
                continue

            if (
                feasibility_result.status
                == FeasibilityStatus.FEASIBLE
            ):
                feasible_indices.append(path_index)
            elif (
                feasibility_result.status
                == FeasibilityStatus.INFEASIBLE
            ):
                infeasible_indices.append(path_index)
            else:
                unknown_indices.append(path_index)

        return (
            tuple(feasible_indices),
            tuple(infeasible_indices),
            tuple(unknown_indices),
        )

    @staticmethod
    def _extract_path_lines(
        path: ExecutionPath,
    ) -> frozenset[int] | None:
        """
        Path metadata geçerliyse pozitif satır kümesini döndürür.

        Metadata yoksa veya elle oluşturulmuş path geçersiz satır değerleri
        içeriyorsa ``None`` döndürülür. Böylece servis erişilemezlik kararı
        vermek yerine güvenli biçimde ``UNRESOLVED`` sonucuna yönelir.
        """
        if not path.has_node_metadata:
            return None

        normalized_lines: set[int] = set()

        for line_number in path.line_numbers:
            if line_number is None:
                continue

            if (
                isinstance(line_number, bool)
                or not isinstance(line_number, int)
                or line_number < 1
            ):
                return None

            normalized_lines.add(line_number)

        return frozenset(normalized_lines)

    @staticmethod
    def _validate_coverage_result(
        coverage_result: FunctionCoverageResult,
    ) -> None:
        if not isinstance(
            coverage_result,
            FunctionCoverageResult,
        ):
            raise TypeError(
                "coverage_result bir FunctionCoverageResult "
                "örneği olmalıdır."
            )

    @staticmethod
    def _validate_paths(
        paths: tuple[ExecutionPath, ...],
    ) -> None:
        if not isinstance(paths, tuple):
            raise TypeError(
                "paths bir ExecutionPath tuple'ı olmalıdır."
            )

        if not paths:
            raise ValueError(
                "paths boş olamaz."
            )

        if any(
            not isinstance(path, ExecutionPath)
            for path in paths
        ):
            raise TypeError(
                "paths yalnızca ExecutionPath "
                "nesneleri içermelidir."
            )

    @staticmethod
    def _validate_feasibility_results(
        feasibility_results: tuple[PathFeasibilityResult, ...],
    ) -> None:
        if not isinstance(feasibility_results, tuple):
            raise TypeError(
                "feasibility_results bir PathFeasibilityResult "
                "tuple'ı olmalıdır."
            )

        if not feasibility_results:
            raise ValueError(
                "feasibility_results boş olamaz."
            )

        if any(
            not isinstance(
                result,
                PathFeasibilityResult,
            )
            for result in feasibility_results
        ):
            raise TypeError(
                "feasibility_results yalnızca "
                "PathFeasibilityResult nesneleri içermelidir."
            )

        if any(
            not isinstance(
                result.status,
                FeasibilityStatus,
            )
            for result in feasibility_results
        ):
            raise TypeError(
                "feasibility_results içindeki status alanları "
                "FeasibilityStatus değeri olmalıdır."
            )

    @staticmethod
    def _validate_path_result_count(
        *,
        paths: tuple[ExecutionPath, ...],
        feasibility_results: tuple[PathFeasibilityResult, ...],
    ) -> None:
        if len(paths) != len(feasibility_results):
            raise ValueError(
                "paths ve feasibility_results aynı "
                "uzunlukta olmalıdır."
            )

    @staticmethod
    def _validate_max_visits(
        max_visits_per_node: int,
    ) -> None:
        if (
            isinstance(max_visits_per_node, bool)
            or not isinstance(max_visits_per_node, int)
        ):
            raise TypeError(
                "max_visits_per_node bir integer olmalıdır."
            )

        if max_visits_per_node < 1:
            raise ValueError(
                "max_visits_per_node en az 1 olmalıdır."
            )

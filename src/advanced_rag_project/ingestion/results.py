from dataclasses import asdict, dataclass


@dataclass
class IngestionStats:
    """
    Statistics collected during a directory ingestion run.
    """

    scanned: int = 0
    new: int = 0
    unchanged: int = 0
    modified: int = 0
    deleted: int = 0
    duplicates: int = 0
    failed: int = 0

    @property
    def total_processed(self) -> int:
        """
        Return the total number of documents that
        received an ingestion decision.

        This includes:

        - new
        - unchanged
        - modified
        - duplicates
        - failed
        """

        return (
            self.new
            + self.unchanged
            + self.modified
            + self.duplicates
            + self.failed
        )

    @property
    def has_errors(self) -> bool:
        """
        Return True if one or more documents failed
        during ingestion.
        """

        return self.failed > 0

    def to_dict(self) -> dict[str, int]:
        """
        Convert the statistics object into a dictionary.
        """

        return asdict(self)
from dataclasses import dataclass


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

    def print_summary(self) -> None:
        """
        Print a human-readable ingestion summary.
        """

        print("\n")
        print("=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)

        print(
            f"Documents scanned : {self.scanned}"
        )

        print(
            f"New               : {self.new}"
        )

        print(
            f"Unchanged         : {self.unchanged}"
        )

        print(
            f"Modified          : {self.modified}"
        )

        print(
            f"Deleted           : {self.deleted}"
        )

        print(
            f"Duplicates        : {self.duplicates}"
        )

        print(
            f"Failed            : {self.failed}"
        )

        print("=" * 60)
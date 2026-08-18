from __future__ import annotations

import sys

import export_manual_review_set as base


SCENARIO_GROUPS = {
    "missing_screws": {"SCREW"},
    "all_empty": {"SCREW"},
    "excess_screws": {"EMPTY"},
    "good": {"SCREW", "EMPTY"},
}


def _requested_scenarios(argv: list[str]) -> list[str]:
    values: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--scenario" and i + 1 < len(argv):
            values.append(argv[i + 1].strip().lower())
            i += 2
            continue
        if arg.startswith("--scenario="):
            values.append(arg.split("=", 1)[1].strip().lower())
        i += 1
    return [v for v in values if v]


def main() -> None:
    requested = _requested_scenarios(sys.argv[1:])

    # Manual review is intentionally scoped by business scenario:
    #   missing_screws -> only Sxx (positions that MUST contain a screw)
    #   all_empty      -> only Sxx (all required screws are missing)
    #   excess_screws  -> only Exx (positions that MUST stay empty)
    #   good           -> both Sxx and Exx
    # This prevents irrelevant ROI groups from being shown as "false positives"
    # while the user is reviewing one defect category at a time.
    if requested:
        allowed_groups: set[str] = set()
        unknown: list[str] = []
        for scenario in requested:
            groups = SCENARIO_GROUPS.get(scenario)
            if groups is None:
                unknown.append(scenario)
            else:
                allowed_groups.update(groups)

        if unknown:
            raise SystemExit(
                "Unsupported scenario(s) for scoped review: "
                + ", ".join(unknown)
                + ". Supported: "
                + ", ".join(sorted(SCENARIO_GROUPS))
            )

        original_group_of = base.group_of

        def scoped_group_of(roi_id: str) -> str:
            group = original_group_of(roi_id)
            return group if group in allowed_groups else "OTHER"

        base.group_of = scoped_group_of
        print(
            "Scenario ROI policy: "
            + ", ".join(requested)
            + " -> "
            + ", ".join(sorted(allowed_groups)),
            flush=True,
        )
    else:
        print(
            "No --scenario supplied: reviewing both Sxx and Exx ROI groups.",
            flush=True,
        )

    base.main()


if __name__ == "__main__":
    main()

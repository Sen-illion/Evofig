"""Stable CLI command classification used by help and navigation tooling."""

PRODUCT_COMMANDS = (
    "paper-evolve-editable",
    "paper-evolve-validate",
    "paper-evolve-corpus",
    "paper-evolve-session",
    "paper-evolve-continuous",
    "paper-evolve-patches",
    "experience-archive",
    "creator-rules",
    "paper-evolve-review",
)
EXPERIMENTAL_COMMANDS = (
    "judge-data",
    "judge",
)
MAINTENANCE_COMMANDS = (
    "doctor",
    "inspect-pdf",
    "benchmark",
    "validate",
    "contract-crosscheck",
    "presentations-qa",
    "analyze-reference-figures",
    "select-reference-exemplars",
    "learn-reference-style",
    "adopt-undelivered-decks",
    "figure-edit",
)

COMMAND_CATEGORIES = {
    **{name: "product" for name in PRODUCT_COMMANDS},
    **{name: "experimental" for name in EXPERIMENTAL_COMMANDS},
    **{name: "maintenance" for name in MAINTENANCE_COMMANDS},
}

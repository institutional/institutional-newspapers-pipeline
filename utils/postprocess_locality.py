_LOCALITY_CORRECTIONS: dict[tuple[str, str], str] = {
    ("Boston", "NY"): "MA",
    ("Lawrence", "NH"): "MA",
}


def postprocess_locality(
    city: str,
    state: str,
    country: str,
) -> tuple[str, str, str, bool]:
    """Corrects known city/state mismatches."""
    if not city or not state:
        return (city, state, country, False)

    corrected_state = _LOCALITY_CORRECTIONS.get((city, state))
    if corrected_state is not None:
        return (city, corrected_state, country, True)

    return (city, state, country, False)

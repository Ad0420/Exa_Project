"""Reading Exa /search response bodies: the results array and each result's company entity."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyEntity:
    id: str | None  # Exa's stable company entity identifier, when present
    properties: Mapping[str, object]


def search_results(body: Mapping[str, object]) -> list[object]:
    """The `results` array of a /search response body, or [] if it is missing or malformed."""
    results = body.get("results")
    return results if isinstance(results, list) else []


def company_entity(result: object) -> CompanyEntity | None:
    """The result's first company entity, if it has one with a `properties` object."""
    if not isinstance(result, Mapping):
        return None
    entities = result.get("entities")
    for entity in entities if isinstance(entities, list) else []:
        if isinstance(entity, Mapping) and entity.get("type") == "company":
            properties = entity.get("properties")
            if isinstance(properties, Mapping):
                identifier = entity.get("id")
                return CompanyEntity(
                    identifier if isinstance(identifier, str) else None, properties
                )
    return None


def company_properties(result: object) -> Mapping[str, object] | None:
    """The `properties` of the result's first company entity, if any."""
    entity = company_entity(result)
    return entity.properties if entity is not None else None

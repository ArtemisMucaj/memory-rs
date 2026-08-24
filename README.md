# memory-rs

Long-term memory for coding assistants: import sessions, extract durable
facts about entities, recall by hybrid semantic + keyword + recency search.

The store is small on purpose. Each memory is one subject–predicate–object
fact with a human-readable statement; entities are resolved by exact match on
a normalized name; recall ranks RRF over cosine similarity, keyword match,
and `recorded_at` so newer memories carry more weight.

## Releases

The macOS release binary is signed with a Developer ID and notarized by Apple,
so it runs without a Gatekeeper exception.

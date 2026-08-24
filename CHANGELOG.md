# Changelog

## [0.2.4](https://github.com/ArtemisMucaj/memory-rs/compare/memory-rs-v0.3.2...memory-rs-v0.2.4) (2026-08-24)


### ⚠ BREAKING CHANGES

* Memory no longer carries predicate, subject or object; the repository interfaces are reshaped around facts and entities.

### Features

* **dream:** gate auto-import on namespace membership and creation date ([#19](https://github.com/ArtemisMucaj/memory-rs/issues/19)) ([471f882](https://github.com/ArtemisMucaj/memory-rs/commit/471f882b158836f87fa17e8d59d0bafa965e729a))
* memory-rs — standalone long-term memory (core, CLI, TUI, API/MCP) ([3a740d8](https://github.com/ArtemisMucaj/memory-rs/commit/3a740d899690cbc417ce2cc9eed32a996baf7aed))
* serve the memory graph over HTTP and MCP, with a resume briefing ([#6](https://github.com/ArtemisMucaj/memory-rs/issues/6)) ([52686e4](https://github.com/ArtemisMucaj/memory-rs/commit/52686e455e872bc531c21e74741fcf3ea1e539d2))
* simplify memory model to facts + entities only ([#27](https://github.com/ArtemisMucaj/memory-rs/issues/27)) ([ee4d2b5](https://github.com/ArtemisMucaj/memory-rs/commit/ee4d2b5d312b0040070d84a0240c4e93ec3b48ad)), closes [#24](https://github.com/ArtemisMucaj/memory-rs/issues/24)


### Bug Fixes

* drop the Windows release target so macOS/Linux assets ship ([#8](https://github.com/ArtemisMucaj/memory-rs/issues/8)) ([fa04485](https://github.com/ArtemisMucaj/memory-rs/commit/fa04485a2558ee83e35360991845eb5d2fa55bf1))
* **ingestion:** scope prefetch to globals when the project is unknown ([#21](https://github.com/ArtemisMucaj/memory-rs/issues/21)) ([baf7c30](https://github.com/ArtemisMucaj/memory-rs/commit/baf7c3004c6f99ec7dfb91f20646ce54a070deab))
* note that the macOS release binary is signed and notarized ([#11](https://github.com/ArtemisMucaj/memory-rs/issues/11)) ([25b3c06](https://github.com/ArtemisMucaj/memory-rs/commit/25b3c06b98a1b34b5dd057214c70ca7a4ed08699))
* re-release to publish the notarized macOS binary ([#14](https://github.com/ArtemisMucaj/memory-rs/issues/14)) ([189c144](https://github.com/ArtemisMucaj/memory-rs/commit/189c144ddafd0d8069d3c5993a6a083558d46a08))
* re-release to publish the notarized macOS binary ([#17](https://github.com/ArtemisMucaj/memory-rs/issues/17)) ([4707adc](https://github.com/ArtemisMucaj/memory-rs/commit/4707adcc22d15d6841a218853926c14451f7bdad))
* **sessions:** scope an imported session to the project it ran in ([#25](https://github.com/ArtemisMucaj/memory-rs/issues/25)) ([7c08da3](https://github.com/ArtemisMucaj/memory-rs/commit/7c08da377547bec37d1efda8fb0b994062fa7730))

## [0.3.2](https://github.com/ArtemisMucaj/memory-rs/compare/v0.3.1...v0.3.2) (2026-08-24)


### Bug Fixes

* **sessions:** scope an imported session to the project it ran in ([#25](https://github.com/ArtemisMucaj/memory-rs/issues/25)) ([7c08da3](https://github.com/ArtemisMucaj/memory-rs/commit/7c08da377547bec37d1efda8fb0b994062fa7730))

## [0.3.1](https://github.com/ArtemisMucaj/memory-rs/compare/v0.3.0...v0.3.1) (2026-08-13)


### Bug Fixes

* **ingestion:** scope prefetch to globals when the project is unknown ([#21](https://github.com/ArtemisMucaj/memory-rs/issues/21)) ([baf7c30](https://github.com/ArtemisMucaj/memory-rs/commit/baf7c3004c6f99ec7dfb91f20646ce54a070deab))

## [0.3.0](https://github.com/ArtemisMucaj/memory-rs/compare/v0.2.4...v0.3.0) (2026-08-08)


### Features

* **dream:** gate auto-import on namespace membership and creation date ([#19](https://github.com/ArtemisMucaj/memory-rs/issues/19)) ([471f882](https://github.com/ArtemisMucaj/memory-rs/commit/471f882b158836f87fa17e8d59d0bafa965e729a))

## [0.2.4](https://github.com/ArtemisMucaj/memory-rs/compare/v0.2.3...v0.2.4) (2026-08-05)


### Bug Fixes

* re-release to publish the notarized macOS binary ([#17](https://github.com/ArtemisMucaj/memory-rs/issues/17)) ([4707adc](https://github.com/ArtemisMucaj/memory-rs/commit/4707adcc22d15d6841a218853926c14451f7bdad))

## [0.2.3](https://github.com/ArtemisMucaj/memory-rs/compare/v0.2.2...v0.2.3) (2026-08-05)


### Bug Fixes

* re-release to publish the notarized macOS binary ([#14](https://github.com/ArtemisMucaj/memory-rs/issues/14)) ([189c144](https://github.com/ArtemisMucaj/memory-rs/commit/189c144ddafd0d8069d3c5993a6a083558d46a08))

## [0.2.2](https://github.com/ArtemisMucaj/memory-rs/compare/v0.2.1...v0.2.2) (2026-08-04)


### Bug Fixes

* note that the macOS release binary is signed and notarized ([#11](https://github.com/ArtemisMucaj/memory-rs/issues/11)) ([25b3c06](https://github.com/ArtemisMucaj/memory-rs/commit/25b3c06b98a1b34b5dd057214c70ca7a4ed08699))

## [0.2.1](https://github.com/ArtemisMucaj/memory-rs/compare/v0.2.0...v0.2.1) (2026-08-04)


### Bug Fixes

* drop the Windows release target so macOS/Linux assets ship ([#8](https://github.com/ArtemisMucaj/memory-rs/issues/8)) ([fa04485](https://github.com/ArtemisMucaj/memory-rs/commit/fa04485a2558ee83e35360991845eb5d2fa55bf1))

## [0.2.0](https://github.com/ArtemisMucaj/memory-rs/compare/v0.1.0...v0.2.0) (2026-08-03)


### Features

* serve the memory graph over HTTP and MCP, with a resume briefing ([#6](https://github.com/ArtemisMucaj/memory-rs/issues/6)) ([52686e4](https://github.com/ArtemisMucaj/memory-rs/commit/52686e455e872bc531c21e74741fcf3ea1e539d2))

## 0.1.0 (2026-07-27)


### Features

* memory-rs — standalone long-term memory (core, CLI, TUI, API/MCP) ([3a740d8](https://github.com/ArtemisMucaj/memory-rs/commit/3a740d899690cbc417ce2cc9eed32a996baf7aed))

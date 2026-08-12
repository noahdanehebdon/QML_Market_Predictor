# Data sources and rights

The project retrieves data only through user-configured provider access. It does
not redistribute provider-derived market data. Raw and processed data are local
research inputs and are excluded from version control.

Users are responsible for their provider subscriptions, permissions, terms, and
applicable exchange agreements. Model output is not investment advice, and no
provider sponsors or endorses this project.

## Alpaca market data

Daily equity and `SPY` bars may be retrieved through Alpaca's Market Data API
with the user's credentials and entitlements. Do not redistribute or publicly
display Alpaca-derived data unless your agreements permit it.

## Federal Reserve Board data

Selected interest-rate and industrial-production series come from public Board
of Governors releases, including H.15 and G.17. They are used only to derive
macroeconomic research features.

## Bureau of Labor Statistics data

CPI and unemployment series may be retrieved through the BLS Public Data API.
Record the retrieval date when producing research because published series can
be revised.

## SEC EDGAR data

Ticker mappings, submissions metadata, and XBRL company facts may be retrieved
from SEC EDGAR. Automated requests must follow SEC fair-access guidance. Set a
descriptive `SEC_USER_AGENT` containing project and contact information, request
only what is needed, and respect the repository's pacing controls.

## Storage policy

Provider responses, processed datasets, fitted models, predictions, reports,
databases, and experiment stores are ignored by Git. Automated data snapshots
and retraining artifacts are stored in a private, user-controlled Cloudflare R2
bucket. See [data versioning](data_versioning.md) for identity, integrity, and
retention controls.

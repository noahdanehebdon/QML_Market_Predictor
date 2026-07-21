# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's private
vulnerability reporting feature. Do not open a public issue containing a
credential, exploit, private endpoint, or provider-derived dataset.

Include the affected file or workflow, reproduction steps, likely impact, and
any suggested mitigation. Reports will be acknowledged as soon as practical.

## Secrets and data

This repository does not require committed credentials. Local secrets belong in
`.env`; GitHub-hosted credentials belong in repository Actions secrets. Market
data, model artifacts, reports, databases, and experiment stores are generated
locally or stored privately and are excluded from Git.

If a credential is exposed, revoke it at the provider immediately. Removing it
from the latest commit is not sufficient because Git history and Actions logs
may retain earlier values.

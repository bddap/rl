# Working on rl

Edit by subtraction: resolve a problem by deleting code; a tactical patch over a symptom is not accepted. One implementation per thing, never two alive.

The history of a codebase should not be encoded in or inferrable from the codebase itself. Systems at rest should look as if they were designed this way from the start. Exceptions exist, including git history and changelogs. Implications include immediate refactors to simplify and remove old patterns, and immediate sweeps and cleanups of code and documentation included with each modification.

Question designs and propose better ones. Large refactors are welcome; there is no
stable API to preserve. Fix problems you find and unit-test behavior where practical.
This is a learning project; explain disagreements directly, with dry humor welcome.

Delete code comments; keep only a why the code cannot show.

## Checks

Run Cargo inside [shell.nix](shell.nix). Before submitting code:

- `cargo fmt --check`
- `cargo clippy --quiet --all-targets -- --deny warnings`
- `cargo test -q -- --test-threads=2`

[test-map.json](test-map.json) selects covering suites. A test runs in the map or is
deleted; do not use `#[ignore]`.

## Boundaries

Keep this project independent. Reference other projects only as declared, versioned
dependencies, exposing names and versions rather than internals. Give shared services
neutral project-owned names. Exclude deployment-specific paths, addresses, service
or queue names, credentials, camera frames and private renders. Before landing,
inspect the diff for undeclared project references and deployment details.

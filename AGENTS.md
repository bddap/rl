# AGENTS.md

./shell.nix for dev dependencies

Question designs. Treat the larger stucture of this project as mutable, don't assume the prexisting code is right.
Don't be afraid to make large refactors, we have no stable api to maintain. Unit test what you can. Delete stuff.

Avoid unnessesary code comments. Delete them, even.

See something wrong, fix it.

Your human is knowlegable, but not infinitely so. Question him, teach him, this project is for fun and learning after all. He will be especially appretiative when you calls him out on his designs. Push back on plans, he'll appretiate when you suggest a better solution than what he asked for. Dry sass is appretiated.

# Pre-submition checks
- `cargo fmt --check`
- `cargo clippy --quiet -- --deny warnings`
- `cargo test -q`

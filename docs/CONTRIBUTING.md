# Team Delegation Guidelines & Commit Conventions

## Branch Strategy
All feature work must be done on isolated feature branches off `main`:

- **Charan (FL Engine)**: `feat/fl-core`
- **Om (Distributed Systems & Infra)**: `feat/infra-nodes`
- **Meerja (Privacy & Cryptography)**: `feat/crypto-engine`
- **Manav (Full-Stack UI)**: `feat/dashboard-ui`
- **Anudeep (ML & Data Pipeline)**: `feat/ml-pipeline`

## Commit Conventions
Follow Conventional Commits format:

```text
<type>(<scope>): <short summary>

[optional body]
```

### Allowed Types:
- `feat`: A new feature
- `fix`: A bug fix
- `build`: Changes that affect the build system or external dependencies
- `chore`: Infrastructure, setup, or workflow changes
- `docs`: Documentation only changes
- `test`: Adding missing tests or correcting existing tests

### Examples:
- `build(data): add MONAI and PyTorch dependencies with base dataset directory structure`
- `feat(fl): scaffold base Flower client-server package structure and configs`
- `chore(infra): initialize hospital directories, env templates, and gitignore`
- `build(privacy): set up TenSEAL environment dependencies and privacy package`
- `feat(ui): initialize Vite + React dashboard with Tailwind configuration`

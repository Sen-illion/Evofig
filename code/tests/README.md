# Test layers

- `unit/`: deterministic functions and local file/PPTX behavior.
- `integration/`: multi-component workflows using mocks, placeholders, or temporary files.
- `e2e/`: real direct paper-to-PPTX acceptance cases. This layer is intentionally empty until benchmark fixtures and production credentials are defined.

Run all local tests with:

```powershell
python -m unittest discover -s tests -q
```

"""One adapter per model. Each hides its model's integration quirks (docs/ml-models-spec.md §0)
and returns plain Python values, so the rest of the agent does not care which runtime is underneath."""

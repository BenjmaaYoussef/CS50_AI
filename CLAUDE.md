# CLAUDE.md — Project-wide instructions for Claude Code

## Code comments

Every code cell must have comments. No exceptions.

Comments must:
- Be written in simple, plain English
- Explain **what the code is doing and why**, not just restate the syntax
- Be written as if explaining to someone seeing the code for the first time

This is a hard requirement because team members must be able to defend any line of code during the presentation Q&A.

**Example of a bad comment:**
```python
# fit the model
model.fit(X_train, y_train)
```

**Example of a good comment:**
```python
# Train the model on the training set only.
# We never pass X_test here — the model must not have seen test data during training,
# otherwise our evaluation metrics would be misleadingly optimistic.
model.fit(X_train, y_train)
```

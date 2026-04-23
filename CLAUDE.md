# CLAUDE.md — Project-wide instructions for Claude Code

## Code comments

Every code cell must have comments. No exceptions.

Comments must:
- Be written in simple, plain English
- Follow a step-by-step style: "Step 1: do X, Step 2: do Y"
- Say what the code does, not why at length

This is a hard requirement because team members must be able to defend any line of code during the presentation Q&A.

**Example of a bad comment:**
```python
# fit
model.fit(X_train, y_train)
```

**Example of a good comment:**
```python
# Step 1: train the model on the training data
# Step 2: use the trained model to predict labels for the test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

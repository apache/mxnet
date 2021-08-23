---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'Bug, needs triage'
assignees: ''

---
## Description
(A clear and concise description of what the bug is.)

### Error Message
(Paste the complete error message. Please also include stack trace by setting environment variable `DMLC_LOG_STACK_TRACE_DEPTH=100` before running your script.)

## To Reproduce
(If you developed your own code, please provide a short script that reproduces the error. For existing examples, please provide link.)

### Steps to reproduce
(Paste the commands you ran that produced the error.)

1.
2.

## What have you tried to solve it?

1.
2.

## Environment

***We recommend using our script for collecting the diagnostic information with the following command***
`curl --retry 10 -s https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/diagnose.py | python3`

<details>
<summary>Environment Information</summary>

```
# Paste the diagnose.py command output here
```

</details>

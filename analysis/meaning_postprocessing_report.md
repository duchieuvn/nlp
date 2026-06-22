# Equation Meaning Postprocessing Report

Generated from the postprocessed equation-meaning audit records.
Flag counts overlap because one record may fail multiple checks. Flagged
meanings are reported for review but preserved in postprocessed data.

## Summary

| Metric | Count |
| --- | ---: |
| Papers | 54 |
| Meaning records | 353 |
| Non-empty phrases | 353 |
| Empty meanings | 0 |
| Flagged meanings | 126 |
| Changed records | 227 |

Phrase length in natural-language words: minimum 1, median 5, maximum 124.

## Selection Strategies

| Strategy | Records |
| --- | ---: |
| `science_head_window` | 159 |
| `no_reliable_phrase` | 126 |
| `subject_before_introduction` | 28 |
| `existing_phrase` | 23 |
| `named_complement` | 9 |
| `derived_object` | 6 |
| `active_context_object` | 2 |

## Flag Reasons

`Records` counts unique flagged records affected by the reason. `Candidates` counts every candidate that failed validation.

| Reason | Records | Candidates |
| --- | ---: | ---: |
| `missing_science_head` | 67 | 165 |
| `symbol_definition_sentence` | 50 | 50 |
| `contains_finite_or_context_verb` | 48 | 71 |
| `incomplete_or_invalid_edge` | 45 | 89 |
| `unbalanced_delimiters` | 26 | 42 |
| `too_short` | 19 | 28 |
| `no_candidates_generated` | 8 | 0 |
| `math_dominated` | 6 | 6 |
| `contains_equation_reference` | 4 | 4 |
| `too_long` | 2 | 2 |

## Representative Flagged Cases

### `incomplete_or_invalid_edge`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2401.00059` | `2` | We substitute Eq. ( 2 ) into Eq. ( 1 ), and obtain |
| `2401.00059` | `6` | We substitute such superposition into equation ( 5 ) and obtain |
| `2401.00059` | `7` | To simplify this, we use Eq. ( 3 ) and obtain |

### `contains_finite_or_context_verb`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2401.00059` | `2` | We substitute Eq. ( 2 ) into Eq. ( 1 ), and obtain |
| `2401.00059` | `7` | To simplify this, we use Eq. ( 3 ) and obtain |
| `2401.11088` | `5` | We quantize each float in R , J , \vec{a} , and \vec{b} separately, then perform the multiplications in equation 5 for each gate application in the circuit (Algorithm 1 ). |

### `missing_science_head`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2401.00059` | `2` | We substitute Eq. ( 2 ) into Eq. ( 1 ), and obtain |
| `2401.00059` | `6` | We substitute such superposition into equation ( 5 ) and obtain |
| `2401.00059` | `7` | To simplify this, we use Eq. ( 3 ) and obtain |

### `too_short`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2401.11088` | `5` | We quantize each float in R , J , \vec{a} , and \vec{b} separately, then perform the multiplications in equation 5 for each gate application in the circuit (Algorithm 1 ). |
| `2404.07802` | `12` | Clearly, with our approach we aim to obtain predictions that at least outperform the accuracy of the trivial estimation: |
| `2405.00701` | `6` | In this paper, we consider the underlying asset prices \vec{S}(t)=(S_{1}(t),\cdots,S_{d}(t)) in the Black-Scholes (BS) model described by the following stochastic differential equation |

### `no_candidates_generated`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2402.14235` | `1` | and the action I_{\mathrm{E}} is the Euclidean Einstein-Hilbert term plus a bare cosmological constant \Lambda_{0} , |
| `2402.14235` | `2` | and the action I_{\mathrm{E}} is the Euclidean Einstein-Hilbert term plus a bare cosmological constant \Lambda_{0} , |
| `2410.17702` | `1` | We can characterise squeezed states by the quadrature fluctuations \expectationvalue*{(\Delta\hat{X}_{\theta})^{2}}=\expectationvalue*{\hat{X}_{% \theta}^{2}}-\expectationvalue*{\hat{X}_{\theta}}^{2} where \hat{X}_{\theta}=[\hat{a}\exp(-... |

### `symbol_definition_sentence`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2403.03204` | `4` | where B(T_{1},T_{2})=B_{A_{1}F_{1}}(T_{1})\,{\oplus}\,B_{A_{2}F_{2}}(T_{2}) denotes the collective action of the two beam splitters with B_{ij}(T) being the beam splitter operation given in Eq. ( 17 ) of the Appendix A . |
| `2404.07802` | `9` | where the total evolution time T is discretized into \frac{T}{\delta t} Trotter steps, -2J\delta t=\phi , and 2h(t)\delta t=\theta . |
| `2404.07802` | `10` | where the total evolution time T is discretized into \frac{T}{\delta t} Trotter steps, -2J\delta t=\phi , and 2h(t)\delta t=\theta . |

### `unbalanced_delimiters`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2405.00701` | `6` | In this paper, we consider the underlying asset prices \vec{S}(t)=(S_{1}(t),\cdots,S_{d}(t)) in the Black-Scholes (BS) model described by the following stochastic differential equation |
| `2407.16814` | `2` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |
| `2407.16814` | `3` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |

### `math_dominated`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2405.00701` | `6` | In this paper, we consider the underlying asset prices \vec{S}(t)=(S_{1}(t),\cdots,S_{d}(t)) in the Black-Scholes (BS) model described by the following stochastic differential equation |
| `2407.16814` | `2` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |
| `2407.16814` | `3` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |

### `too_long`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2407.16814` | `2` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |
| `2407.16814` | `3` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |

### `contains_equation_reference`

| Paper | Equation | Original selected evidence |
| --- | --- | --- |
| `2407.16814` | `2` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |
| `2407.16814` | `3` | From equations ( 2 ) and ( 3 ), we can define A_{j}=\sum_{i=0}^{n-1}a_{i}(\beta\xi^{j})^{i}\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ \text{for }\leavevmode\nobreak\ \leavevmode% \nobreak\ \leavevmode\nobreak\ j=0,1... |
| `2407.16814` | `5` | From equations ( 2.1 ) and ( 5 ), we get the required result. |

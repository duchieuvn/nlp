# Symbol Meaning Rejection Report

Generated from empty symbol-meaning records and reconstructed BM25 evidence.
A rejected symbol has no reliable extractive definition; the symbol itself
is still preserved in the final dataset.

## Summary

| Metric | Count |
| --- | ---: |
| Papers | 54 |
| Extracted symbols | 1897 |
| Defined symbols | 202 |
| Empty definitions | 1695 |
| Empty percentage | 89.35% |

Retrieval uses `bm25` with top-k `12` candidates.

## Rejection Reasons

| Reason | Symbols | Percentage of empty definitions |
| --- | ---: | ---: |
| `no_supported_definition_pattern` | 1444 | 85.19% |
| `no_retrieved_alias` | 251 | 14.81% |

## Representative Rejected Cases

### `no_supported_definition_pattern`

| Paper | Equation | Symbol | Best retrieved evidence |
| --- | --- | --- | --- |
| `2401.00059` | `1` | `u_n_double_dot` | For all the atoms but those on the interface ( u^{L}_{0} and u^{R}_{0} ) we write down the motion equation in the form Equation (1): \ddot{u}_{n}=-\beta(u_{n}-u_{n-1})-\beta(u_{n}-u_{n+1}) where we omitted the side index L or R , since t... |
| `2401.00059` | `1` | `u_n` | For all the atoms but those on the interface ( u^{L}_{0} and u^{R}_{0} ) we write down the motion equation in the form Equation (1): \ddot{u}_{n}=-\beta(u_{n}-u_{n-1})-\beta(u_{n}-u_{n+1}) where we omitted the side index L or R , since t... |
| `2401.00059` | `1` | `u_n_1` | For all the atoms but those on the interface ( u^{L}_{0} and u^{R}_{0} ) we write down the motion equation in the form Equation (1): \ddot{u}_{n}=-\beta(u_{n}-u_{n-1})-\beta(u_{n}-u_{n+1}) where we omitted the side index L or R , since t... |

### `no_retrieved_alias`

| Paper | Equation | Symbol | Best retrieved evidence |
| --- | --- | --- | --- |
| `2401.00059` | `2` | `A_bar` | As it is well known [ 53 ] , the solution to such equation is (line denotes the complex conjugation) |
| `2401.00059` | `5` | `u_0_sup_R_double_dot` | The solution in such form will automatically satisfy motion equations for all atoms (Eq. 1 ) but those interfacial ones (Eq. 4 ). |
| `2401.00059` | `5` | `u_1_sup_R` | The masses of atoms in the media are m^{L} and m^{R} , and interatomic distances are a^{L} and a^{R} (Fig. 1 ). |

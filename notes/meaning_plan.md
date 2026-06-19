# Equation Meaning Extraction Plan

## Goal

The script `source/step4_eqn_meaning.py` finds a short name for each equation.
For example, it may extract names such as:

- `covariance matrix of two uncorrelated thermal modes`
- `Wigner characteristic function of the TMST state`
- `four-mode entangled state`

The script uses local text processing. It does not send prompts to a generative AI
model.

## Input and Output

The input is:

```text
data/3_equations.json
```

Each equation entry contains a `surrounding_text.window` field. The position of
the equation in this text is shown by:

```text
[EQUATION]
```

The output is:

```text
data/4_equation_meanings.json
```

The output keeps the original equation data. It adds the extracted name to the
`meaning` field and adds details to `audit-trail`.

## Step 1: Load the Language Model

The script loads the local SciSpaCy model:

```text
en_core_sci_scibert
```

This model splits the text into tokens, sentences, and noun phrases. The script
also tells its tokenizer to keep `[EQUATION]` as one token.

This is not the custom equation NER model used by `step4_b.py`.

## Step 2: Prepare the Context

The complete context can be too long for SciBERT. The script first keeps up to
40 words before and 40 words after `[EQUATION]`.

It also removes some LaTeX commands and characters that are not useful for
finding a name.

If the text is still too long, the script tries smaller windows:

```text
40, 25, 15, or 8 words on each side
```

## Step 3: Look for an Anchor Phrase

The first and strongest method looks for common phrases before the equation.
Examples include:

- `can be written as`
- `is given by`
- `takes the form`
- `defined as`
- `reads`
- `satisfies`
- `represented by`

For this text:

```text
The four-mode entangled state is represented by [EQUATION]
```

the anchor is `is represented by`. The script checks the noun phrases before
the anchor and can select:

```text
four-mode entangled state
```

An anchor result gets `high` confidence in the audit record.

## Step 4: Expand the Noun Phrase

Sometimes the useful name contains a prepositional phrase. The script can
expand a noun phrase through these words:

```text
of, for, with, in, on
```

For example, it can keep the complete name:

```text
covariance matrix of two uncorrelated thermal modes
```

Expansion stops at verbs, punctuation, conjunctions, another anchor phrase, or
the `[EQUATION]` marker.

## Step 5: Try Fallback Methods

If no anchor produces a reliable name, the script tries two fallback methods.

### Nearby Noun Phrase

It looks for a noun phrase in the same sentence, close to `[EQUATION]`. The
phrase must contain a scientific head word such as:

```text
equation, function, matrix, model, operator, relation, state
```

The phrase must end no more than 12 tokens before the marker.

### Context Verb

If the first fallback fails, the script looks for context verbs such as:

```text
compute, denote, evaluate, obtain, represent, yield
```

It then accepts a scientific noun phrase in the same sentence up to 24 tokens
before the marker.

Both fallback methods use `medium` confidence.

## Step 6: Filter Bad Candidates

Before accepting a name, the script rejects candidates that:

- are empty or longer than 12 words;
- look like a numbered reference, such as `Eq. (4)`;
- contain too many symbols;
- contain only one generic word, such as `equation` or `state`, unless an anchor
  clearly supports that word.

It also removes punctuation and simple leading words such as `the`, `a`, `our`,
and `this`.

## Step 7: Rank Candidates

When several noun phrases are available, the script ranks them. It prefers a
candidate that:

1. contains a scientific head word;
2. is in the same sentence as `[EQUATION]`;
3. was expanded with a useful prepositional phrase;
4. has a useful amount of detail;
5. is close to the equation or anchor.

## Step 8: Save the Result and Audit Record

When a candidate is found, the script writes it to `meaning`. It also records:

- the extraction method;
- the candidate;
- the confidence level;
- the strategy used;
- the anchor phrase, when one was used;
- a short source-text excerpt.

Example:

```json
{
  "meaning": "four-mode entangled state",
  "audit-trail": [
    {
      "meaning_extraction": {
        "method": "SciSpaCy/SciBERT dependency/noun-chunk extraction from surrounding_text.window",
        "candidate": "four-mode entangled state",
        "confidence": "high",
        "strategy": "anchor",
        "trigger": "represented by",
        "trigger_type": "anchor_pattern"
      }
    }
  ]
}
```

If no reliable candidate is found, `meaning` stays empty. The audit record uses
`blank` confidence and explains why no name was selected.

## Run the Script

From the project root, run:

```bash
python source/step4_eqn_meaning.py
```

The script prints the number of equations visited, filled, and left blank.

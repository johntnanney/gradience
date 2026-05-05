# AI-Language Scan Checklist

Reference document for pre-submission AI-language audits. The companion
script `scripts/ai_language_scan.sh` automates the lexical and phrasal
checks; this document captures the broader pattern catalogue, including
items the script can't easily check (sentence-length variance,
register consistency, stylometric signals).

**Calibration principle.** This is a clustering signal, not a blacklist.
Most patterns below have legitimate uses. The signal comes from
clustering — when several appear together in the same document, or when
one appears repeatedly in the same author voice, that is the tell. A
single instance of any item below is rarely diagnostic; six or more
hits across categories in a single document is.

**Domain calibration.** Some patterns flagged below are normal in
specific domains. "Robust" in a statistics paper is technical
vocabulary, not filler. "Embark" in a maritime history paper is
literal, not metaphor. Calibrate by domain when reading scan output;
the script reports hits, not verdicts.

---

## 1. Lexical tells — overrepresented words

Words measurably overrepresented in academic and professional writing
after 2023, per the "Delving into ChatGPT usage in academic writing"
arXiv study and follow-up analyses.

The companion scanner (`scripts/ai_language_scan.sh`) splits this
category into two sub-layers because some words have legitimate
technical use in measurement-theory, statistics, and ML domains:

**Layer 1a — high-precision AI tells** (counted toward cluster score):
delve, scaffolding, ascertain, underscore, paramount, tapestry,
showcasing, pivotal, realm, meticulous, illuminate, unveil, elevate.
These have little legitimate technical use in academic prose; almost
every instance is replaceable by a more direct word.

**Layer 1b — domain-overlap candidates** (reported informationally,
not counted toward cluster score): foster, harness, cultivate, empower,
optimize, streamline, encompass, embody, manifest, traverse. Each of
these has substantive technical use in some domains: "harness" appears
in `lm-evaluation-harness` (a tool name); "manifest" is a psychometric
term ("manifest variable" vs "latent variable"); "optimize" / "optimizing"
appears legitimately in optimization, budget allocation, and statistical
estimation contexts; "encompass" / "encompasses" is technical in
measurement-universe writing. The scanner reports per-hit so the user
can judge case-by-case whether usage is technical or AI-tell.

**Other overused vocabulary worth watching** (not in the scanner's
core layers but flagged in the broader literature): intricate,
multifaceted, holistic, comprehensive, robust (in non-statistical
contexts), leverage, navigate (figurative), myriad, nuanced, profound,
transformative, vibrant, rich (as in "rich tapestry"), embark.

**Hype vocabulary:** game-changer, revolutionize, groundbreaking,
cutting-edge, state-of-the-art, paradigm shift, seismic shift,
transformative, unprecedented, disruptive, next-generation.

**Performative-engagement vocabulary:** compelling, fascinating,
intriguing, captivating, riveting, enthralling, profound, deeply (as
intensifier — "deeply concerning," "deeply important").

## 2. Phrasal patterns — multi-word constructions

### Framing pivots

- "That's not X, it's Y" / "X is not just Y; it's Z" / "It's not merely X, it's Y" / "X isn't just about Y, it's about Z"
- "This is where X comes in" / "Enter X" / "X enters the picture"
- "The beauty of X is..." / "The power of X lies in..." / "The elegance of X is..."
- "The short answer is..." / "the long answer is..." / "TL;DR" (in non-meta contexts)
- "Here's the thing" / "Here's the catch" / "Here's the kicker"
- "To summarize" / "In summary" / "In essence" / "Essentially" / "At its core" / "At the end of the day" / "When all is said and done"
- "Let's dive in" / "Without further ado" / "Buckle up"
- "What sets X apart is..." / "What makes X unique is..." / "What's noteworthy about X is..."

### Substitution patterns (replacing direct verbs with longer alternatives)

- "serves as," "stands as," "marks," "represents" (in place of "is")
- "plays a [crucial/pivotal/key/central] role" (in place of "matters" or being specific)
- "speaks volumes about" (in place of saying what it shows)
- "is a testament to" (in place of "demonstrates" or "shows")
- "sheds light on" / "shed new light on" (in place of "shows" or "explains")
- "underscore the importance of" (in place of "emphasize" or being direct)

### Setting/framing openers

- "In today's [adjective] world/landscape/era/age"
- "In the realm of X" / "In the world of X" / "In the domain of X"
- "When it comes to X" / "As far as X is concerned"
- "More and more" / "Increasingly" / "In recent years" / "In the past decade"
- "Picture this:" / "Imagine that..."

### Hedge/qualification clusters

- "It's worth noting that..." / "It is worth noting that..."
- "It's important to note that..." / "It is important to note that..."
- "It's worth mentioning that..." / "It bears mentioning..."
- "It goes without saying" / "Needless to say"
- "It can be argued that..."
- "Some might say..." / "One might say..."
- "It is generally accepted that..." / "It is widely held that..."

## 3. Sentence-level rhetorical structures

**Three-beat cadences.** "Fast, efficient, and reliable." Watch for
*adjective, adjective, and adjective* or *noun, noun, and noun* where
the three items are roughly synonymous or fill a similar slot. A list
of three substantively different things is fine; three near-synonyms
is the tell.

**Mid-sentence rhetorical questions.** "But now? You won't be able to
unsee this." / "The solution? It's simpler than you think." / "Why does
this matter? Because..." Used as throat-clearing rather than as genuine
questions. A genuine rhetorical question that carries the argument
forward is fine; a question used purely for cadence is the tell.

**Anaphora — repeated sentence openings.** Three or more parallel
sentences starting with the same word or short construction. "It is X.
It is Y. It is Z." / "We need X. We need Y. We need Z."

**Anadiplosis — chained repetition.** "We need precision. Precision
requires discipline. Discipline requires pre-registration." Heavily
overrepresented in motivational and explanatory AI prose.

**The "not just X. It is Y" pivot.** "X is not merely Y. It is Z."
Recognizable LLM cadence even when each individual instance does
substantive work. Two instances in the same document is a clustering
signal.

**"Not only X but also Y."** Overused by AI, especially when X and Y
are roughly synonymous or weakly distinguishing. A strong "not only X
but also Y" introduces something genuinely additional; an AI-pattern
"not only X but also Y" is filler.

**Em-dash overuse.** Heavy use of `---` for parenthetical asides,
especially when commas would work. AI overuses em-dashes both as
parentheticals and as sentence-internal pivots ("This isn't just
X --- it's Y"). One or two em-dashes per page is normal; six per page
is a tell.

**Sentence-length uniformity (burstiness).** AI-generated text tends to
have lower sentence-length variance than human writing. Human writers
cluster long descriptive sentences with short punchy ones; AI
distributes them more evenly. Hard to grep for, but worth scanning
visually.

## 4. Connective and transitional patterns

### Overused paragraph-opening connectives

- "Furthermore," / "Moreover," / "Additionally," — fine occasionally; clustered, they are a tell
- "However," / "Nevertheless," / "Notwithstanding,"
- "Consequently," / "Therefore," / "Thus," / "Hence,"
- "In conclusion," / "In summary," / "To conclude," / "All in all," / "Ultimately,"

### Adverb-heavy sentence openers

- "Importantly," / "Crucially," / "Notably," / "Significantly,"
- "Indeed," / "Indeed ---"
- "Specifically," / "Particularly," / "Especially,"
- "Effectively," / "Essentially," / "Fundamentally,"

### The "Of course," tic

- "Of course, X" used as throat-clearing rather than as genuine concession.

## 5. Verbosity substitutions

AI writing systematically substitutes longer, more abstract Latinate
words for shorter concrete ones.

| AI-leaning | Direct |
|---|---|
| utilize | use |
| facilitate | help |
| elucidate | explain |
| ascertain | determine, find out |
| endeavor | try |
| commence | start, begin |
| terminate | end |
| approximately | about |
| in order to | to |
| due to the fact that | because |
| in light of the fact that | because |
| owing to the fact that | because |
| for the purposes of | for |
| in spite of the fact that | although |
| with respect to | about, regarding |
| in the event that | if |
| at this point in time | now |
| at the present time | now |
| a substantial number of | many |
| a vast majority of | most |
| sufficient quantity of | enough |
| has the ability to | can |
| is in possession of | has |

## 6. Academic-methodology-specific patterns

### Boilerplate paper openers and closers

- "This study/paper aims to..." / "The aim of this study is..." / "This work seeks to..."
- "We will explore..." / "We will examine..." / "We will discuss..."
- "Let us examine..." / "Let us consider..."
- "It is interesting to note..."
- "The findings suggest that..." (when followed by a hedge)
- "Significant implications for..." / "Far-reaching implications for..."
- "A growing body of evidence suggests..."
- "Recent advances have shown..."
- "It is widely accepted that..."
- "This study contributes to the literature by..."

### Methodology overclaims

- "rigorous analysis" / "rigorous methodology"
- "comprehensive analysis" (when it's just thorough)
- "novel approach" / "innovative methodology"
- "valuable insights"
- "robust findings" (in non-statistical contexts)
- "promising results"

### Limitations boilerplate

- "While X has limitations, it nonetheless contributes Y..."
- "Future research could explore..."
- "More work is needed to..."
- "Additional research could investigate..."

## 7. Closing and opening boilerplate (conversational AI tells)

These show up in collaborative AI-assisted writing when the assistant's
voice leaks into the final document.

### Closings

- "I hope this helps!"
- "Let me know if you have any questions"
- "Feel free to reach out"
- "Looking forward to hearing your thoughts"

### Openings

- "Certainly!" / "Absolutely!" / "Great question!"
- "I'd be happy to..."
- "That's a great point."
- "Let me break this down for you..."

## 8. Stylometric signals (require analysis tooling beyond grep)

These are diagnostic but not catchable by simple regex.

**Sentence-length variance.** Compute the standard deviation of
sentence lengths in a paragraph. Human writing tends to have higher
variance; AI writing tends to cluster around an average. A document
where every paragraph has sentences of similar length (say, all 15-25
words) is suspicious.

**Word entropy / vocabulary breadth.** AI writing tends to use a
smaller range of vocabulary even in technical domains. If a document
uses "important" twelve times where a human writer might rotate through
"important," "central," "load-bearing," "critical," "key," etc., that
is a stylometric signal.

**Adjective and adverb density.** AI writing has higher rates of
adjectives, adpositions, auxiliary verbs, subordinating conjunctions,
and verbs (per the 2025 stylometric study cited in references).
Concrete nouns and active simple verbs are underrepresented.

**Citation patterns (academic-specific).** AI-generated academic
writing often uses citations as decoration rather than as load-bearing
reference. Watch for citations that appear in the bibliography but
aren't engaged substantively in the prose, or for clusters of three
citations after a generic claim ("Recent work has explored these
issues (X, Y, Z)") where the three works don't actually share the
diagnosis being attributed.

**Paragraph length uniformity.** Like sentence length, but at the
paragraph scale. AI tends to produce paragraphs of similar length (say,
4-6 sentences); human academic writing tends to mix dense methodology
paragraphs with shorter linking paragraphs.

**Consistent register.** AI tends to maintain a uniform register
throughout. Human writers tend to have moments of higher informality,
occasional sharp asides, varying levels of technicality. A document
that maintains exactly the same register for 19 pages without any
variation in tone is suspicious.

## 9. How to use this checklist

**Build a layered scan, not a single grep.** Start with the
highest-precision tells (specific overused words and exact phrases),
then move to medium-precision (verbosity substitutions, connective
overuse), then to low-precision stylometric signals (sentence-length
variance) only if the high-precision scans are inconclusive. The
companion script `scripts/ai_language_scan.sh` implements this layered
structure.

**Cluster, don't blacklist.** A document with one "delve" in it isn't
AI-written; a document with "delve" + "underscore" + "paramount" +
heavy em-dashes + uniform sentence lengths is. Score by cluster
density.

**Domain-calibrate.** Some patterns that are AI tells in general
writing are normal in your domain. "Robust" in a statistics paper is
fine; "robust" in a marketing brochure is filler. The N134 audit
caught this correctly --- the single "robust" hit in that paper was
technical statistical usage, not AI filler.

**Watch for the "polished but generic" signal.** Even when no
individual phrase is flagged, AI writing tends to have a quality of
being smooth without being specific. If you finish a paragraph and
can't summarize what concrete claim it made, that is a tell even if no
lexical patterns flagged.

**Two-pass review.** First pass with grep/regex against the lexical
and phrasal patterns (the script does this). Second pass with a human
reader explicitly asking "does any sentence feel like it could appear
in any document on this topic?" --- generic sentences are AI tells
even when no specific word triggers.

**Repetition is the strongest signal.** A single instance of any
pattern can be coincidence. The same pattern appearing twice or more
in the same document is the tell. The N134 paper had two instances of
the "not merely X. It is Y" pivot --- a single instance would have
been forgivable; two instances clustered as an AI signature.

## 10. References

- *Stylometric analysis of AI-generated texts: a comparative study of ChatGPT and DeepSeek*, Cogent Arts & Humanities, 2025. https://doi.org/10.1080/23311983.2025.2553162
- *Stylometric comparisons of human versus AI-generated creative writing*, Humanities and Social Sciences Communications, Nature, 2025. https://doi.org/10.1038/s41599-025-05986-3
- Liang, Yang, et al. *Delving into ChatGPT usage in academic writing through excess vocabulary*. arXiv:2406.07016. https://arxiv.org/abs/2406.07016
- Wikipedia: *Signs of AI writing*. https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- *Unveiling ChatGPT text using writing style*, Heliyon (PMC), 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11231544/

---
title: "Health & Social Care in Hansard: a topic-pilot brief"
subtitle: "The pipeline, the pilot test, and the substantive checks we now need"
date: "July 2026"
geometry: "a4paper, margin=2.4cm"
fontsize: 11pt
linkcolor: "RoyalBlue"
urlcolor: "RoyalBlue"
header-includes:
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\footnotesize}
  - \setlength{\tabcolsep}{4pt}
---

## 1. The goal

We want to measure what Parliament has said about health and social care (H&SC)
across the full Hansard record, 1803 to the present. For every speech we want to
know two things: whether it substantively engages H&SC, and if so which aspects of
it the speaker raises. With that in place we can trace how the volume and the mix
of H&SC debate shift over two centuries, by era and by topic. The obstacle is
scale. The record runs to millions of speeches, far more than anyone can read, so
the measurement has to be done by machine and then trusted. This note describes
the pipeline we have built to do that, the pilot we ran to test it, and the
substantive questions we now need domain input on before scaling up.

## 2. The main pipeline, step by step

The production pipeline has four stages. Each exists for a reason worth stating,
because the reasons are also where the method could go wrong.

1. Narrow the corpus to plausible candidates. The large majority of Hansard never
   touches H&SC, so running a language model on every speech would be wasteful. In
   production this stage will be a semantic search over sentence embeddings,
   retrieving speeches close in meaning to H&SC whatever their exact wording, and
   that retrieval step will itself need validation. For now we use a broad keyword
   net (NHS and public-health terms for the modern era; Poor Law, workhouse,
   sanitary-reform and epidemic-disease terms for the nineteenth century) as a
   deliberate placeholder: it is crude, but it lets us build and test the rest of
   the pipeline without waiting on the retrieval work. Either way this stage decides
   only what gets looked at, never the answer, and it is deliberately
   over-inclusive. It leaves a few hundred thousand candidate speeches rather than
   millions.

2. Read each candidate with a language model. For every surviving speech the model
   answers three questions: does this speech substantively discuss H&SC (yes or
   no); if so, which sub-topics does it raise; and a supporting verbatim quote.
   The judgement needs to weigh meaning in context, which a keyword cannot do. The
   sub-topics are free text, written by the model rather than chosen from a fixed
   list, because the aim is to discover what is actually discussed rather than to
   score speeches against categories we set in advance.

3. Aggregate the free-text sub-topics into a taxonomy. The models describe the
   same idea in many wordings, so to count topics we group same-meaning phrases
   automatically and label each group. Section 5 explains how, and why this step
   is more delicate than it looks.

4. Turn the labelled topics into measures over time. The payoff is prevalence: how
   much of H&SC debate each topic occupies, and how that changes across the 220
   years.

Stage 2 sends the model a short instruction, an output-format request, and the
speech. The core elements are reproduced verbatim below; the pilot varies their
surface form (Section 3) but not their meaning. The topic definition is carried
inside the instruction, which makes it easy to overlook as a piece of wording. It
is in fact the most consequential setting in the pipeline, and Section 5 shows
what happens when it changes.

Instruction (one of two paraphrases):

> Determine whether the parliamentary speech below substantively discusses health
> and social care — that is, UK health and social care policy: the NHS, public and
> mental health, adult social care, care homes, and the people who provide or rely
> on that care — rather than merely mentioning it in passing. If it does, identify
> the specific sub-topics of health and social care that it discusses. Describe
> each sub-topic in your own words as a short phrase (a few words). Give at most 5
> sub-topics. Also provide the single most relevant verbatim quotation from the
> speech as supporting evidence.

Output format (the structured condition):

> Respond with a single JSON object and nothing else, with exactly these keys:
> "mentions_topic" (true or false), "subthemes" (an array of short free-text
> phrases, empty if mentions_topic is false), and "evidence_quote" (a string,
> empty if mentions_topic is false).

The other output condition gives no format instruction at all, leaving the model
to answer in free text. When present, an optional system message adds a role: "You
are an expert analyst of health and social care policy in the United Kingdom, with
a deep understanding of parliamentary debate."

## 3. The pilot: a test before scaling

Running stages 1 to 4 on the whole corpus is expensive and hard to undo, so we
first tested the approach on a sample of 270 speeches, stratified across eras and
speech lengths. The pilot adds one thing the production run will not use: a
robustness harness. Instead of reading each speech once, we read it many ways and
check that the answer is driven by the speech rather than by incidental choices of
model or wording. If the answers move a lot under variation, the production
numbers are not yet trustworthy.

Concretely, each speech was read by four models (three production candidates plus
one larger reference model) under eight wordings of the same request, varying the
role framing, the paraphrase, and the output format. That is 32 independent reads
per speech. Two results frame everything that follows:

- On the yes/no presence question the reads agree with each other about 82% of the
  time, which is solid but short of unanimous.
- The share of speeches counted as being about H&SC ranges from 34% to 81%
  depending on prompt and model, averaging around 51%. Much of that swing comes
  from the output format: asking for free-text output puts presence at about 55%,
  while asking for structured output puts it at about 43%. The headline rate is
  therefore sensitive to prompt design, which is one reason the substantive checks
  below matter.

## 4. What the models found: the topic map

The topic map below is built from the configuration we now expect to ship: the
expert healthcare-then-social-care definition, under structured (JSON) output, no
role framing, and no five-topic cap. Across the 270 speeches that arm produced
1,412 distinct sub-topic phrases (1,673 emissions). Grouped by meaning, the most
common topics are shown below. Both modern NHS-era
themes and nineteenth-century themes such as Poor Law, sanitation and housing
appear. This coverage should be read with the sampling in mind: the keyword net
was deliberately built to include both modern and historical H&SC vocabulary, so
the presence of both is partly a consequence of how the sample was seeded rather
than independent evidence about the balance of the corpus. What the map does show
is what the models surface as sub-topics once a speech is placed in front of them,
and that is what the rest of this note examines.

![Most common H&SC topics under the expert HC→SC definition with structured output (share of speeches raising each; count in brackets). Labels are the clusters' automatically chosen medoid phrases, not hand-edited.](fig_top_topics.png)

Each row below is a machine-discovered topic, that is, a cluster of same-meaning
phrases, ranked by how many of the 270 speeches raise it. The "Phr." column counts
how many distinct wordings the models used for that one idea. The label in each row
is the cluster's medoid: the single emitted phrase closest to the cluster's centre
in meaning-space, chosen automatically rather than written by us. Nothing here is
hand-edited, which is why some labels read awkwardly or are oddly cased (for
example "support for elderly people" or "housing conditions"). Naming the topics
well is one of the domain judgements we are asking for.

| # | Merged topic | Speeches | Phr. | Example sub-themes the models wrote |
|:--|:----------------------|:---------|----:|:----------------------------------------------|
| 1 | NHS funding and planning | 51 (19%) | 112 | NHS funding/performance; NHS resource management; NHS funding |
| 2 | Poor relief and pauperism | 32 (12%) | 79 | Poor Law relief; Poor law/pauperism; Poverty and the Poor Law |
| 3 | Public health | 31 (11%) | 41 | Public Health Policy; Public Health Measures; Public health infrastructure |
| 4 | Social welfare provision | 23 (9%) | 33 | Social welfare support; Social services funding; Social welfare governance |
| 5 | Health service funding | 19 (7%) | 29 | Healthcare funding; public health funding; public financing of health services |
| 6 | Access to healthcare | 19 (7%) | 27 | Access to healthcare services; Care and treatment access; Healthcare access |
| 7 | Carer recognition and support | 18 (7%) | 46 | Social care funding; Care for carers; Public funding for care |
| 8 | support for elderly people | 13 (5%) | 37 | Care for pensioners; Old age support; Elderly care |
| 9 | housing conditions | 12 (4%) | 30 | Social housing; Public health housing; Housing standards |
| 10 | Mental health services | 12 (4%) | 29 | Hospital provision for mental health; Mental health provision; mental health facilities |
| 11 | Hospital governance | 11 (4%) | 24 | Hospital management and administration; Hospital administration and efficiency; hospital management |
| 12 | Public involvement in healthcare decision-making | 11 (4%) | 26 | Decentralisation of healthcare decisions; Public involvement in healthcare; Patient involvement in healthcare |
| 13 | Disease prevention | 11 (4%) | 12 | preventive healthcare; Preventative healthcare; Preventive Care |
| 14 | Overcrowding and living conditions | 10 (4%) | 15 | Overcrowded housing; Barrack conditions; Poor living conditions |
| 15 | Sanitation | 10 (4%) | 19 | Sanitary conditions; Sanitary conditions in public buildings; Sewage and sanitation |
| 16 | Disease outbreak control | 10 (4%) | 24 | Epidemic control; Epidemic response; Infectious disease response |
| 17 | Hospital discharge planning | 10 (4%) | 17 | hospital care; Hospital discharge; Hospital preservation |
| 18 | transparency in public health inquiries | 9 (3%) | 14 | Transparency of public health bodies; Public trust in healthcare delivery; transparency in health decision-making |
| 19 | Nursing workforce shortages | 8 (3%) | 19 | Workforce crisis in healthcare; Workforce crisis in health; Nursing workforce |
| 20 | Healthcare regulatory agency oversight | 8 (3%) | 14 | Healthcare agencies; Health commission; Department of Health oversight |
| 21 | Children's residential care | 8 (3%) | 19 | Children in care; child protection services; protection of young children in care settings |
| 22 | vaccination | 8 (3%) | 22 | vaccination and immunisation; immunisation; vaccination policy |
| 23 | Healthcare charges | 7 (3%) | 19 | Private fees for medical practitioners in pay beds; Patient charges/costs; Pay beds and private fees in public hospitals |
| 24 | Support for victims | 7 (3%) | 8 | Support for the vulnerable; support for recovery; Support for vulnerable immigrants |
| 25 | Integration of health and social care services | 6 (2%) | 20 | Patient-centred services; Integrated care; Patient-centred care services |
| 26 | Food poverty | 6 (2%) | 7 | Food costs and access; Food supply shortages; Food poverty and food bank support |
| 27 | Social worker training and support | 6 (2%) | 13 | Care worker training; Social worker training; Health professional training and support |
| 28 | tuberculosis treatment | 6 (2%) | 13 | TB care and treatment; Tuberculosis control; TB treatment |
| 29 | Medical aid provision | 6 (2%) | 10 | Medical aid; Healthcare supply chain; Medical Relief |
| 30 | Independence of the medical profession | 6 (2%) | 16 | Doctor compensation and retirement; Medical profession independence; Doctor recruitment to under-doctored areas |
| 31 | Well-being as a guiding principle | 6 (2%) | 7 | Well-being as guiding principle; Mental and physical well-being in care; Physical well-being and basic living standards |
| 32 | Government funding allocation | 6 (2%) | 6 | Public funding allocation; Funding and expenditure oversight; funding and budget allocation |
| 33 | Parliamentary scrutiny of care regulations | 5 (2%) | 7 | Reform of healthcare and social care regulatory bodies; parliamentary scrutiny of health regulations; Timing of health and social care legislation |
| 34 | Learning disability nursing | 5 (2%) | 15 | nursing care; Learning disability nursing shortages; Mental health nursing |
| 35 | Health and working conditions | 5 (2%) | 11 | Health of working life; Workplace health risks; health impacts of poor working conditions |
| 36 | Mental capacity legislation | 5 (2%) | 11 | Mental health and capacity law; Mental Capacity; Mental capacity and decision-making |
| 37 | Health visitor roles | 5 (2%) | 9 | Health visitor role; Health visitor public health role; Service personnel healthcare access |
| 38 | disease diagnosis | 5 (2%) | 8 | disease detection and diagnosis; Diagnosis of illness; health diagnostics |
| 39 | Taxation on low-income households | 5 (2%) | 8 | Taxation and its impact on low-income groups; Local taxation for relief; Local fiscal burden and taxation |
| 40 | Access to social support | 5 (2%) | 4 | Social integration; Social needs prioritization; Social clubs |

A long tail of smaller topics sits below this list and grows increasingly
specific: 152 clusters in total, of which
50 are raised by four or more speeches. Ranks
41 onward are in the appendix.

## 5. Where the sub-topics come from, and why aggregation is delicate

The topic map is not a neutral readout of what Parliament said. It is shaped at
several points by choices we made, each of which the pilot suggests we should
revisit with domain input.

**The definition sets the era profile.** The presence and era findings below
compare several construct wordings; the topic map in Section 4 uses the expert
healthcare-then-social-care wording we expect to ship. The original pilot
definition, quoted in Section 2, names the NHS, adult social care and care homes. Those are institutions that did not exist for
the first century and a half of the record, so the wording risks reading the
nineteenth century as quiet about health simply because it lacked the vocabulary.
We tested this by re-running the same speeches, models and formats against an
era-neutral rewrite that describes the function rather than the institution: the
health of the population and the care of people who are sick, injured, disabled,
elderly or destitute, however that care is provided and paid for. The effect is
large, and concentrated exactly where the concern predicted. Under the current
definition the share of speeches judged H&SC climbs from 33% before 1900 to 62%
after 1948; under the era-neutral wording it runs from 54% to 67%, and the rise
across the record falls from 29 to 13 percentage points (Figure 2). The two
definitions agree on 82% of individual reads, and the disagreements run almost
entirely one way: the era-neutral wording adds speeches, in 23% of pre-1900 reads
but only 7% of post-1948 ones. On this evidence a good part of the apparent
growth in health debate after the NHS is an artefact of how we asked the question.

![Share of speeches judged to be about H&SC, by era, under four construct definitions. Neither is known to be correct; the point is that the choice of wording, not the record, sets much of the slope.](fig_definition_era.png)

This does not establish that the era-neutral wording is the right one. Reading the
speeches it newly admits shows both kinds of case. Some are plain misses by the
current definition: an 1868 exchange in which the Poor Law Board sets standards
for the diet, lodging and work of vagrants is unmistakably the Victorian care
system, and the current wording scored it as H&SC in only one read out of eight.
Others look like over-reach: an 1856 debate about which body should hold powers
under an agricultural statistics bill was counted as H&SC by seven of eight reads
because the Poor Law Board happens to be named in it, which is precisely the
passing mention the instruction rules out. That second case is worth stating
plainly, because it is the same failure as before rather than a new one. The old
wording triggered on the token NHS; the new one triggers on tokens like Poor Law
Board. Changing the definition moved which century the anchoring flatters, but did
not stop the models keying on institution names instead of on whether care is the
subject of the speech. Choosing between the two wordings, and repairing whichever
we keep so that neither trap fires, is a domain judgement and not something a
further run can settle.

**An expert definition of both healthcare and social care lands close to
current, not to era-neutral.** Domain experts supplied separate sentences for
healthcare ("primarily concerned with the prevention, diagnosis and treatment of
illness") and social care ("practical support to people who need assistance with
everyday living…"). Because the construct of interest is speeches about *both*,
the two sentences were concatenated and run as a new definition arm under the
same shipping default (no role, no cap, both formats, all models). Two orders
were tested: healthcare then social care, and social care then healthcare. The
headline rates sit next to current rather than next to era-neutral: 51% under
HC→SC and 46% under SC→HC, against 48% for current and 62% for era-neutral.
Matched-cell agreement with current is 88% and 87% respectively; speech-level
majority labels agree with current on 96% and 93% of speeches. The era profile
tells the same story (Figure 2): both expert orders keep a steep post-1948 rise
(+24 and +23 percentage points from pre-1900), close to current (+29) and far
from era-neutral (+13). So replacing the pilot wording with the expert sentences
does not, by itself, undo the NHS-era climb that the earlier definition contrast
flagged as partly artefactual. The order of the two sentences is a real but
secondary effect: HC-first is about five points more inclusive than SC-first on
matched cells (89% agreement; 166 cells yes only under HC→SC versus 62 only under
SC→HC), and only 17 of 270 speeches flip their majority label. Free-text output
is more order-sensitive than structured output; one model (Llama-70B) is almost
order-invariant. If a single expert wording is shipped, HC→SC is the closer twin
of the current pilot definition. The contested speeches under each pair —
expert-order flips, and current versus either expert arm — can be inspected in
`definition_review.html`, which now lets the reader switch comparison presets.

**The expert definition changes who is counted more than what is named.**
Sub-topics were collected on every positive read of the definition arm, so the
maps can be compared directly. Holding format fixed at structured output (the
cleaner stream, and the one we intend to ship), the expert HC→SC map and the
current-definition map agree on twenty of the top twenty-five topics by meaning
(embedding match), and the ranks of those matched topics correlate at Spearman
0.83. The headline families stay put: NHS funding, public health, Poor Law /
pauper relief, social welfare, access to care, mental health, housing and
sanitation. What moves is emphasis at the margin. Under the expert wording,
carers and elderly support rise into the top ten, and disease prevention,
overcrowding and healthcare-regulator oversight enter the top twenty-five; under
the current wording, local governance of health services and health
infrastructure sit higher. Absolute speech counts are a little lower than in the
earlier full-grid map because this comparison uses one definition × one format
rather than eight prompt wordings, but the shape of the taxonomy is stable enough
that swapping the shipping definition does not force a fresh substantive review
of the topic list from scratch — only a check on the handful of families that
shift rank.

**The prompt's output format changes the result.** Each read was asked either for
free-text output or for structured output, and this matters more than expected.
The number of sub-topics per speech is unaffected (about 3.8 either way, often
hitting the cap of five), so the format does not squeeze the count. What it changes
is the wording and the presence rate. Free-text output produced 5,903 distinct
phrasings; structured output produced only 3,639, roughly 40% fewer, because
structured output regularises how the model names the same idea while free text
lets it drift. Free-text output also raised the presence rate (55% versus 43%) and
failed to parse cleanly about 5% of the time, which is the source of the noise
described below. So the raw phrase counts in the table, and how badly topics
fragment, are partly a property of prompt design rather than of the speeches. In
production we will fix a single output format, and that is itself a measurement
decision, not just an engineering one.

**The five-topic cap inflates the count.** The instruction asks for at most five
sub-topics, and we have now tested removing it: the same speeches, models and
formats, run once with the "at most five" sentence and once without. The cap turns
out to work as an anchor, not just a ceiling. With the cap, half of all positive
reads return exactly five sub-topics and the median is five. Without it, that spike
disappears (only 9% land on five), the distribution settles onto a smooth curve
centred on two to three, and the median drops to three (Figure 3). About 14% of
speeches genuinely warrant more than five sub-topics when allowed, up to eighteen,
but those are the minority; the dominant effect is the cap pulling the many
lower-content speeches up towards five. Presence is unaffected (48% either way), so
this is purely about how many sub-topics get counted, not whether a speech is
judged to be about H&SC. The practical implication is that any count-based measure
built on the capped runs is biased upward, so production should drop the cap or set
it far higher.

![Sub-topics returned per positive read, capped versus uncapped, on the same speeches. The cap concentrates half of all reads at exactly five; without it the count follows a smooth low distribution.](fig_cap_effect.png)

**Aggregation depends on one threshold.** Same-meaning phrases are merged by
converting each phrase into a numeric representation of its meaning (an embedding)
and clustering the ones that sit close together, with each cluster labelled by its
most central phrase (its medoid). A single setting controls how readily phrases merge. Set it
loose and unrelated ideas get lumped together; set it tight and one idea fragments
into several topics. We currently keep it tight to avoid false merges, and the
cost of that is visible in the table: NHS funding and planning (row 1) sits apart
from Health service funding (5) and Carer recognition and support / social care
funding (7); Public health (3) sits apart from Disease prevention (13) and
Disease outbreak control (16); and Hospital governance (11), transparency in
public health inquiries (18) and Healthcare regulatory agency oversight (20)
split what could be one governance family. Where these should be merged, and how
broad the final families should be, is a domain judgement rather than a distance
measurement.

**A small share of entries are formatting noise, not content.** From the free-text
parse failures above, a model's answer sometimes leaked into the topic field.
These are a known and fixable side effect, not real findings, but they are worth
seeing:

- Does the speech substantively discuss health and social care? No
- Sub-topics of health and social care discussed: None
- Most relevant verbatim quotation (for context): ...

Taken together, the prevalence numbers depend jointly on the definition, the
output format, the five-topic cap, and the clustering threshold. We can now put
rough sizes on the first two: switching from the current wording to era-neutral
moves the era profile by around 16 percentage points of slope, while swapping in
the expert healthcare-plus-social-care wording leaves that slope largely intact
(+23 to +24 pp rather than +29) and moves the headline rate by only a few points;
presentation order within the expert wording adds about another 5 points. The
output format moves the headline rate by about 12 points under every definition
tested. None of these is settled by the machine, which is precisely why the next
step is substantive rather than technical.

## 6. What we need next

Two pieces of work now need domain input, and they are the reason for this brief.

The first is a substantive review of the current map:

1. Scope. Section 5 shows the definition is the most consequential setting we
   have found, and the pilot cannot resolve it alone: the era-neutral wording
   fails in the opposite direction from the current one, while the expert
   healthcare-plus-social-care wording tracks current closely and does not
   remove the post-1948 climb. Which bound is closer to how the field thinks of
   H&SC, should the expert sentences be preferred as the shipping text (and in
   which order), and how would you word the construct so that neither the NHS
   nor the Poor Law Board pulls a speech in on the strength of being mentioned?
2. Completeness. Is any important H&SC theme missing that Parliament would have
   debated across this period?
3. Correctness. Is any listed topic wrong, or wrongly counted as H&SC?
4. Grouping and grain. How far should the over-split topics in Section 5 be merged,
   and at what level of detail should the final set sit?

The second is designing how the production pipeline will be evaluated, which the
paper will stand or fall on. The pilot only shows that the models agree with each
other, which is consistency, not that they agree with the field, which is validity.
To close that gap we need to build, together:

- A gold standard: a set of speeches hand-labelled by domain experts for presence
  and for sub-topics. This is the reference the whole system is measured against,
  and the definition experiment has already produced an efficient starting set.
  On 159 of the 270 pilot speeches the two definitions reach different verdicts, 92
  of them by a wide margin, and those are the speeches where the boundary is
  actually being decided rather than being obvious. They are laid out in
  `definition_review.html`, an offline page that shows each speech next to what the
  models said under both definitions and records a yes, borderline or no judgement
  with a note, then exports the lot as a spreadsheet. The same page now also covers
  the expert-definition arm: a preset switches the columns to HC→SC versus SC→HC
  (or either expert arm against current), so the 17 majority-label order flips and
  the larger current-versus-expert disagreements can be reviewed the same way.
  Working through the contested set is worth considerably more per speech than
  labelling a random sample, in which most cases are easy and carry little
  information.
- An evaluation protocol we can re-run unchanged whenever a model, prompt, or
  threshold changes: presence scored against the gold labels, and sub-topics
  scored on whether the model's topics match the experts' in meaning.
- A decision, informed by that protocol, on the production settings the pilot has
  shown to matter: the definition, the output format, the sub-topic cap (Section 5
  shows the current five is biasing counts), and the clustering grain.

Any form works for a first pass: notes in the margin, a shared document, or a call.

## Appendix: reference

A. Filter keywords. The placeholder net from Section 2 matches the terms below, as
word stems, in three groups. A speech enters the sample if it matches any of them;
the terms decide only what is looked at, never the answer.

- Modern (NHS and welfare state): social care, care home, domiciliary care,
  residential care, adult social care, care worker, carer, NHS, national health
  service, public health, mental health, health care, healthcare.
- Pre-NHS (Poor Law and sanitary reform): poor law, workhouse, pauper, board of
  guardians, relieving officer, district nursing, lunatic asylum, feeble-minded,
  mental deficiency, friendly society, infirmary, almshouse, sanitary, board of
  health, fever hospital, vaccination, smallpox, cholera, tuberculosis, medical
  officer of health, dispensary.
- Epidemic and pandemic: covid, coronavirus, pandemic, long covid, lockdown,
  furlough, epidemic, influenza, spanish flu, quarantine, outbreak, typhoid,
  typhus, diphtheria, scarlet fever, plague.

B. Prompt, second paraphrase. Section 2 gives the first wording of the
instruction. The pilot also uses this second, meaning-preserving paraphrase:

> Read the parliamentary speech below. Decide if health and social care is a
> substantive subject of the speech and not just a passing reference. Treat health
> and social care as: UK health and social care policy: the NHS, public and mental
> health, adult social care, care homes, and the people who provide or rely on that
> care. When it is a substantive subject, list the particular aspects of health and
> social care it covers, each as a brief free-text label of a few words. Provide no
> more than 5 such labels. Include a short supporting quotation taken word-for-word
> from the speech.

Role and output format vary as in Section 2 (role present or absent, output
structured or free text), giving eight prompt combinations in all.

C. Definition wordings tested. Each is dropped into the same slot in the
instruction, after "substantively discusses health and social care, that is,".
Everything else in the prompt is identical, so the contrast between them is the
definition and nothing else.

> Current: UK health and social care policy: the NHS, public and mental health,
> adult social care, care homes, and the people who provide or rely on that care.

> Era-neutral: the health of the population and the care of people who are sick,
> injured, disabled, elderly or destitute, however that care is provided and paid
> for, whether by the state, local authorities, hospitals, charities, religious
> bodies or families.

> Expert healthcare: Healthcare is primarily concerned with the prevention,
> diagnosis and treatment of illness.

> Expert social care: Social care provides practical support to people who need
> assistance with everyday living because of age, disability, chronic illness or
> other long-term needs. Typical services include residential care, care homes,
> home care, support for carers and services for adults with disabilities.

The expert arm concatenates those two sentences in both orders (HC→SC and SC→HC)
so that order effects can be measured without inventing new wording. Two further
wordings prepared earlier, one narrower (clinical services only) and one broader
(including the social determinants of health such as housing, sanitation and
nutrition), are still available in config but have not been run; they would bracket
the construct so the headline prevalence can be reported as a range rather than a
single number.

D. Extended topic list, ranks 41 onward. This continues the table in Section 4.
Of the 152 machine-discovered topics under the
expert HC→SC / JSON arm, 50 are raised by four or more speeches.

| # | Merged topic | Speeches | Phr. |
|:--|:-----------------------------|:---------|----:|
| 41 | public health guidance | 5 (2%) | 7 |
| 42 | Healthcare policy | 4 (1%) | 4 |
| 43 | impact of environmental hazards on health | 4 (1%) | 8 |
| 44 | Patient safety | 4 (1%) | 5 |
| 45 | NHS medical job opportunities | 4 (1%) | 9 |
| 46 | hospital infrastructure development | 4 (1%) | 16 |
| 47 | Medical education funding | 4 (1%) | 7 |
| 48 | Medication compliance | 4 (1%) | 13 |
| 49 | Basic salary for doctors | 4 (1%) | 10 |
| 50 | Professional accountability in healthcare | 4 (1%) | 8 |
| 51 | Nurse regulation | 3 (1%) | 13 |
| 52 | Disability provision | 3 (1%) | 7 |
| 53 | Maternity care | 3 (1%) | 3 |
| 54 | suicidal ideation | 3 (1%) | 6 |
| 55 | National health standards | 3 (1%) | 6 |
| 56 | emergency care demand | 3 (1%) | 6 |
| 57 | National Insurance contributions | 3 (1%) | 4 |
| 58 | Access to food services | 3 (1%) | 4 |
| 59 | Racial inequality in healthcare | 3 (1%) | 5 |
| 60 | Service Shaping | 3 (1%) | 3 |
| 61 | Asylum funding | 3 (1%) | 7 |
| 62 | Mismanagement in public funds | 3 (1%) | 4 |
| 63 | Clinical decision-making | 3 (1%) | 5 |
| 64 | Crisis intervention | 3 (1%) | 5 |
| 65 | Veterans' health | 3 (1%) | 5 |
| 66 | Community care | 3 (1%) | 5 |
| 67 | Medical research | 3 (1%) | 6 |
| 68 | Bacon rationing | 2 (1%) | 4 |
| 69 | Statutory fees | 2 (1%) | 2 |
| 70 | Building cost reduction | 2 (1%) | 3 |
| 71 | Distribution of health information through clinics and online | 2 (1%) | 4 |
| 72 | role of medical officers of health | 2 (1%) | 3 |
| 73 | Criminal justice - child homicide | 2 (1%) | 2 |
| 74 | population displacement | 2 (1%) | 2 |
| 75 | Professional misconduct investigations | 2 (1%) | 2 |
| 76 | Detention and legal processes | 2 (1%) | 2 |
| 77 | Blood product safety | 2 (1%) | 9 |
| 78 | Friendly societies and financial protection | 2 (1%) | 3 |
| 79 | Housing infrastructure maintenance | 2 (1%) | 3 |
| 80 | GP access | 2 (1%) | 2 |
| 81 | Union insolvency | 2 (1%) | 2 |
| 82 | Housing equipment and supplies | 2 (1%) | 3 |
| 83 | Judicial wellbeing | 2 (1%) | 6 |
| 84 | Impact of education funding reforms | 2 (1%) | 4 |
| 85 | Animal disease control | 2 (1%) | 4 |
| 86 | at-risk groups identification | 2 (1%) | 4 |
| 87 | Alcohol abuse impact | 2 (1%) | 16 |
| 88 | Working class hardship | 2 (1%) | 3 |
| 89 | Postgraduate medical training | 2 (1%) | 6 |
| 90 | Extension of existing care provisions | 2 (1%) | 2 |
| 91 | Impact of Covid-19 | 2 (1%) | 5 |
| 92 | Legal rights of vulnerable individuals | 2 (1%) | 2 |
| 93 | Resident engagement in local decision-making | 2 (1%) | 2 |
| 94 | Sickness benefit | 2 (1%) | 3 |
| 95 | Non-medical clinicians in care decisions | 2 (1%) | 3 |
| 96 | Medical error reporting | 2 (1%) | 5 |
| 97 | Local government finance | 2 (1%) | 2 |
| 98 | HIV/AIDS care and treatment | 2 (1%) | 5 |
| 99 | Stigma in mental health | 2 (1%) | 5 |
| 100 | cholera and diarrhoea treatment | 2 (1%) | 3 |

E. Most common sub-topic phrases, verbatim, before grouping. These show the raw
vocabulary under the same expert HC→SC / JSON arm. Counts are lower than the
clustered table because models rarely reuse an exact string.

| Sub-theme phrase (verbatim) | Speeches | Total emissions |
|---|---:|---:|
| National Health Service | 11 | 15 |
| Poor Law administration | 9 | 11 |
| Disease prevention | 6 | 9 |
| Social welfare provision | 6 | 6 |
| NHS funding | 5 | 9 |
| Healthcare funding | 5 | 6 |
| Public health | 5 | 5 |
| Tuberculosis treatment | 4 | 7 |
| National Health Service funding | 4 | 7 |
| Mental health services | 4 | 7 |
| Workhouse conditions | 4 | 6 |
| Housing conditions | 4 | 4 |
| Mental Health | 4 | 4 |
| Public Health | 4 | 4 |
| Tuberculosis prevention | 3 | 5 |
| Carer support | 3 | 5 |
| Child welfare | 3 | 4 |
| public health | 3 | 4 |
| poor relief | 3 | 3 |
| Poor Law reform | 3 | 3 |

F. Pilot facts. 270 speeches, read by four models under eight prompt wordings each
(32 reads per speech at temperature zero), producing 8,692 distinct sub-topic
phrases and 15,634 total phrase emissions on that core grid. Controlled
experiments matched to it on speech, model and format accompany it: the uncapped
arm behind Section 5's cap finding (2,160 reads), the era-neutral definition arm
(2,160 reads), and the expert-definition arm in both sentence orders (4,320
reads). Every positive read also returned free-text sub-topics; the topic map in
Section 4 is built from the expert HC→SC definition under structured output
(1,412 distinct phrases). The sample spans 1803 to the present. Sub-topics are generated by the models
rather than chosen from a list. All figures come from the current pilot run.

# Writing style — RAGTAG paper

Read this before any writing or editing task in this repo. These rules govern Claude's behavior; the human author owns the prose.

---

## Workflow (read first)

This file's rules apply *inside* the following per-paragraph rhythm. The rhythm comes first; the rules are how Claude behaves at each step.

**Context:** The paper is submitted to ESEM 2026 and double-submitted as a thesis screened by Turnitin AI detection. Time-pressured (one-week internal draft window). The author owns the prose; Claude's job is to support without contaminating the statistical fingerprint.

**Per paragraph:**

1. **Brainstorm together.** Before the author writes, clarify the one message the paragraph must land and the 2--4 facts that support it. Claude proposes structure if useful; author confirms or pushes back. No prose at this step.
2. **Author drafts.** In the author's voice.
3. **Claude checks.** Terse. Revised paragraph with mechanical fixes baked in per §1, plus a change log and flags. No preaching, no rule recitation.
4. **Emergency draft (rare).** If the author is genuinely stuck, Claude produces a draft. The author then rewrites it heavily in their own voice on a separate pass, ideally with the Claude version not visible. Use sparingly — every emergency draft is a Turnitin risk.

**Barred workflow:** Claude does NOT produce a "supervisor's draft" for the author to magpie phrases from. That mode imports a statistical fingerprint Turnitin flags, and the polished phrases the author would naturally cherry-pick are exactly the ones the detector catches.

**Per section (after the section is fully drafted and revised):** the author runs the section through Turnitin via their institutional access and rewrites flagged passages directly. Empirical findings from these runs feed back into this file — patterns Turnitin consistently flags get added to §2 (hard bans) or §4 (patterns to avoid). The author reports what got flagged; Claude proposes updates.

**This file evolves.** When the rhythm breaks down or a new pattern emerges, Claude proposes an update; the author confirms before edits.

---

## 1. Your stance: targeted fixes inside the author's prose

Your default response to a feedback request is **the revised paragraph itself**, with all fixes you judge necessary baked in, followed by a short change log noting what you altered and why. The author then picks what to keep.

**Preservation rules (non-negotiable):**

- Keep the author's word choices unless they are wrong. "Wrong" includes: typos, dictation errors, homophone slips (its/it's, their/there), subject--verb disagreement, tense shifts, missing articles, capitalization errors, comma splices, missing or misplaced punctuation, run-on sentences, factual errors, banned phrases per §2.
- Keep the author's sentence rhythms. If their sentences are 8, 22, 6 words, your revision must not normalize them to 14, 14, 14.
- Keep the author's structural decisions: which claim leads, which is parenthetical, which clauses connect. Do not re-architect.
- Do not add hedges, transitions, or framing the author chose to omit.
- No vocabulary inflation. If the author wrote "uses," do not change it to "leverages," "employs," or "utilizes."

**The change log** appears below the revised paragraph as a numbered list. One line per fix. Mechanical fixes (typos, missing periods) get one word of explanation. Substantive fixes (a clause moved, a claim sharpened, a citation reordered) get one sentence.

**When to refuse the rewrite:**

- The draft has a structural problem that requires conversation, not a fix. Say so and propose the conversation.
- The author asks "polish this" or "make this better" without naming an axis. Ask which axis (length, clarity, hedging, structure).
- The fix would require inventing a number or claim. Flag and stop.

**On request, switch to pure-critique mode** (line-anchored issues without rewrites) — this is the alternate mode, used when the author wants the diagnostic without the diff.

**AI-detection risk this mode introduces:** every revised paragraph drags the prose slightly toward your register. The author should periodically read sections aloud and listen for sentences that sound smoother than the ones around them. Those are likely yours and should be roughed back up.

---

## 2. Hard bans (no exceptions)

These words and phrases are forbidden in any text you produce or suggest. Flag them in the author's drafts; do not silently rewrite.

**Words:**
crucially, pivotal, groundbreaking, novel, cutting-edge, leverage, delve, elucidate, plethora, myriad, landscape, realm, underscore, underpin, navigate (as metaphor), seamless, holistic, paradigm.

**Phrases:**
- "It is worth noting that"
- "It is important to note that"
- "It should be noted that"
- "This is significant because"
- "Through the lens of"
- "In the realm of"
- "A wealth of"
- "Sheds light on"
- "Paves the way for"
- "Stands as a testament to"
- "It is essential to recognize that"

If the author has these in a draft, flag them. Don't delete without asking.

---

## 3. Punctuation

- **No em-dashes (`—`).** Use a period, comma, colon, parentheses, or semicolon. This is the single biggest AI tell.
- **Semicolons only when load-bearing.** Most should be periods.
- **Numerals for comparisons** (k=9, n=3300, F1=0.775). Spell out one through nine only in non-comparative prose.
- **No Oxford-comma policing.** Match what the author uses.

---

## 4. Sentence and paragraph patterns to avoid

- **Tricolons.** "Fast, accurate, and scalable" is suspicious. List two if two is the truth, four if four is. The reflex to round to three is a giveaway.
- **Roadmap sentences at section starts.** Don't open with "In this section, we...". Start with the claim.
- **Summary sentences at paragraph end.** Don't restate the paragraph. End on the strongest concrete sentence.
- **Smooth bridges between paragraphs.** Don't write "Building on this..." or "With this in mind...". Paragraph breaks do the work.
- **Hedging stacks.** Pick one hedge if needed (may, could, appears to). Never two.
- **"Not just X but also Y" / "Not only X but Y".** Reword.
- **Passive without agent.** "It was found" → "We found". Passive is fine when the data is the subject: "The model was trained on N issues."
- **Symmetric paragraph lengths.** A one-sentence paragraph is fine. Don't normalize.
- **Repeated sentence openers.** Vary how clauses begin. If three consecutive sentences start with "The model...", flag it.


## 5. Process rules

1. **Match the surrounding voice.** Read the paragraphs around your edit point. If the author writes short sentences, don't introduce a long one.
2. **Cite analysis docs by path.** When a claim depends on a number, point to the source CSV. Don't invent or paraphrase numbers.
3. **Flag every uncertainty.** "Verify against `docs/analysis/rq3_llm_rescue.csv`." Don't guess.
4. **Don't restructure unless asked.** If the author wrote sections in a particular order, leave the order alone.
5. **One critique pass per request.** Don't pile suggestions across multiple turns; the author needs to apply one round before the next.
6. **Bake every necessary fix into the revised paragraph; do not pile on bonus rewrites.** "Necessary fixes" are: (a) mechanical correctness — typos, dictation errors, homophone slips, grammar errors, punctuation errors, capitalization, comma splices, run-ons; (b) factual errors; (c) banned phrases per §2; (d) undefined acronyms; (e) inconsistencies with the author's own structure; (f) broken citations or LaTeX errors. Stylistic preferences outside those categories belong in the change log as flags, not as silent edits.

---

## 6. Defaults

- **Tables:** pipe-format markdown, right-aligned numeric columns.
- **First person plural.** "We" is standard for SE/ML papers; "the authors" is stilted.
- **No contractions in the body.** "Do not" not "don't" for camera-ready. (Loosen for blog posts and arXiv-only drafts; ask once per target venue.)
- **Acronyms expanded on first use** per section, not just per paper.
- **Figures and tables referenced explicitly.** "Figure 3 shows..." not "as shown above."

---

## 7. Things that are fine, against common advice

- **Sentence fragments for emphasis.** Used sparingly, they wake the reader up.
- **Starting sentences with "But" or "And".** Standard in modern academic writing.
- **Short paragraphs.** Two-sentence paragraphs are fine when the point is two sentences long.
- **Direct first-person.** "We chose k=9" is better than "k=9 was chosen for this study."



## 8. When in doubt

Read the sentence aloud. If you would not say it to a colleague at a whiteboard, rewrite it. If you cannot read it without taking a breath in the middle, shorten it.

If the author asks for something this file forbids, push back once with a one-sentence explanation. If they insist, comply.

---

## 9. AI-detection guardrails

The mode change in §1 (revised paragraphs are the default) makes voice drift the standing risk. These are principles you internalize, not checklists you announce.

**What to keep in mind on every revision:**

- **Burstiness.** Don't flatten sentence-length variance. If the author writes 8/22/6, don't revise to 14/14/14. Restore variance by reshaping content, never by inserting filler.
- **Vocabulary register.** Every author word that survives copyedit stays verbatim. No silent elevation ("uses" → "leverages," "shows" → "demonstrates"). Ever.
- **Idiosyncrasy preservation.** Mild awkwardness, unusual phrasings, mid-sentence asides — keep them unless they fall under §1's "necessary fix" list. They are the author's voice fingerprint.
- **Pattern compliance.** §4 violations the author made get flagged in the change log. §4 patterns the author did NOT make must NOT appear in the revision.

**Diagnostic tools (use when warranted, not always):**

Produce a sentence-length fingerprint (`lengths: original [11, 17, 11] → revised [13, 18, 17]`) when you suspect you flattened the rhythm, or on the author's request. Most paragraphs don't need it.

**Author's standing habits (Claude's reminder, not Claude's task):**

- Read sections aloud. Sentences that sound smoother than their neighbors are likely Claude's; rough them back up.
- Every 3–4 sections, paste prose through an AI detector (GPTZero, Originality, Pangram). Smoke test, not gospel — sustained scores across multiple sections are signal; one-off hits are not.

**The trade is real.** Every necessary fix slightly costs variance, and the longer the writing window, the more the prose drifts toward Claude's register. The principles above are the brake; your read-aloud habit is the audit.

---

## 10. Out of scope

This file does not govern:

- Caption text in figures (terser is fine; rules 4 and 5 still apply).
- Code comments (different conventions).
- Direct quotes from cited work (preserve exactly).
- Reviewer responses (separate register).

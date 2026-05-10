"""Open-coding for the 60-issue invalid-output sample.

Codes (refined after reading all 60):
  CONTINUES-AS-BODY  — model continues generating issue-body-style content
                       (code, logs, stack traces, expected/actual fields)
                       and never converges on a label.
  OFF-TOPIC-LOOP     — repetitive numeric/character loops (e.g. "5555...",
                       repeated dots) — degenerate generation.
  MALFORMED-OUTPUT   — model output is unparseable as a schema label:
                       empty/minimal output, schema-violating values
                       (e.g. "docs"), or extraneous prose mixed with a
                       partial tag.
  CHAIN-OF-THOUGHT   — model reasons in prose rather than emitting a label.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "sample_invalids.csv")

INV_CODES: dict[tuple, tuple[str, str]] = {
    # --- BRAGTAG ---
    ("BRAGTAG", "ansible_ansible", 232): ("MALFORMED-OUTPUT", "Output 'docs'; valid format, label not in schema."),
    ("BRAGTAG", "bitcoin_bitcoin", 86): ("CONTINUES-AS-BODY", "Continues with error message after prefill."),
    ("BRAGTAG", "bitcoin_bitcoin", 233): ("CONTINUES-AS-BODY", "Continues with mocktime/init log lines."),
    ("BRAGTAG", "dotnet_roslyn", 92): ("OFF-TOPIC-LOOP", "Random off-topic text 'Xamarin.iOS reference assemblies'."),
    ("BRAGTAG", "facebook_react", 41): ("CONTINUES-AS-BODY", "Continues with stack trace + GitHub query string field."),
    ("BRAGTAG", "flutter_flutter", 16): ("CONTINUES-AS-BODY", "Continues with breadcrumb data structure."),
    ("BRAGTAG", "flutter_flutter", 23): ("CONTINUES-AS-BODY", "Continues with paint stack trace."),
    ("BRAGTAG", "flutter_flutter", 27): ("CONTINUES-AS-BODY", "Continues with flutter doctor 'No issues found' line."),
    ("BRAGTAG", "flutter_flutter", 72): ("CONTINUES-AS-BODY", "Continues with framework stack trace."),
    ("BRAGTAG", "flutter_flutter", 88): ("OFF-TOPIC-LOOP", "Repetitive '63322931198...' digit loop."),
    ("BRAGTAG", "flutter_flutter", 98): ("OFF-TOPIC-LOOP", "Repetitive '5555...' loop."),
    ("BRAGTAG", "flutter_flutter", 212): ("CONTINUES-AS-BODY", "Continues with iOS Foundation warning."),
    ("BRAGTAG", "kubernetes_kubernetes", 97): ("CONTINUES-AS-BODY", "Continues with /etc/os-release content."),
    ("BRAGTAG", "kubernetes_kubernetes", 241): ("CONTINUES-AS-BODY", "Continues with template's 'plugins and version' section."),
    ("BRAGTAG", "microsoft_TypeScript", 9): ("CONTINUES-AS-BODY", "Continues writing 'Actual/Expected behavior' fields with JSX."),
    ("BRAGTAG", "microsoft_TypeScript", 145): ("MALFORMED-OUTPUT", "Just '<label>' returned, then nothing."),
    ("BRAGTAG", "microsoft_TypeScript", 174): ("CONTINUES-AS-BODY", "Continues writing the proposal's semantics section."),
    ("BRAGTAG", "microsoft_TypeScript", 267): ("CONTINUES-AS-BODY", "Continues rephrasing the user's question."),
    ("BRAGTAG", "microsoft_vscode", 155): ("MALFORMED-OUTPUT", "'Answer: <label>feature</label>' — correct answer; parser failed."),
    ("BRAGTAG", "microsoft_vscode", 205): ("MALFORMED-OUTPUT", "'Answer: <label>question</label>' — correct answer; parser failed."),
    ("BRAGTAG", "opencv_opencv", 61): ("CONTINUES-AS-BODY", "Continues with sprintf deprecation warning."),
    ("BRAGTAG", "opencv_opencv", 211): ("CONTINUES-AS-BODY", "Continues with C++ code snippet."),
    ("BRAGTAG", "opencv_opencv", 270): ("CONTINUES-AS-BODY", "Continues writing 'Issue submission checklist' field."),
    ("BRAGTAG", "opencv_opencv", 273): ("CONTINUES-AS-BODY", "Continues with compiler error + 'Steps to reproduce'."),
    ("BRAGTAG", "tensorflow_tensorflow", 39): ("MALFORMED-OUTPUT", "'6\\n<label>bug</label>' — bug correct; format malformed."),
    ("BRAGTAG", "tensorflow_tensorflow", 66): ("CONTINUES-AS-BODY", "Continues with grpc_server_lib.cc error log."),
    ("BRAGTAG", "tensorflow_tensorflow", 70): ("CONTINUES-AS-BODY", "Continues with TypeError trace for VariableSpec."),
    ("BRAGTAG", "tensorflow_tensorflow", 131): ("CONTINUES-AS-BODY", "Continues with TileT InvalidArgumentError."),
    ("BRAGTAG", "tensorflow_tensorflow", 146): ("MALFORMED-OUTPUT", "Output 'documentation' (picked from 'Documentation Bug')."),
    ("BRAGTAG", "tensorflow_tensorflow", 272): ("CONTINUES-AS-BODY", "Continues with python imports."),

    # --- RAGTAG ---
    ("RAGTAG", "ansible_ansible", 206): ("CONTINUES-AS-BODY", "Continues with python version line."),
    ("RAGTAG", "bitcoin_bitcoin", 46): ("CONTINUES-AS-BODY", "Continues with cookie/http log lines."),
    ("RAGTAG", "bitcoin_bitcoin", 274): ("MALFORMED-OUTPUT", "'Answer: <label>bug</label>' — model emitted a label; parser failed."),
    ("RAGTAG", "dart-lang_sdk", 250): ("CONTINUES-AS-BODY", "Continues with build output and dart paths."),
    ("RAGTAG", "facebook_react", 178): ("MALFORMED-OUTPUT", "'Answer: <label>feature</label>' — correct answer; parser failed."),
    ("RAGTAG", "facebook_react", 224): ("MALFORMED-OUTPUT", "Just '<label>' returned."),
    ("RAGTAG", "flutter_flutter", 4): ("MALFORMED-OUTPUT", "Just '.' returned."),
    ("RAGTAG", "flutter_flutter", 88): ("CONTINUES-AS-BODY", "Continues with EXCEPTION CAUGHT framework trace."),
    ("RAGTAG", "flutter_flutter", 98): ("OFF-TOPIC-LOOP", "Repetitive '5555...' loop."),
    ("RAGTAG", "flutter_flutter", 215): ("CONTINUES-AS-BODY", "Continues with flutter run -v build output."),
    ("RAGTAG", "flutter_flutter", 230): ("CONTINUES-AS-BODY", "Continues with android.permission.INTERNET XML."),
    ("RAGTAG", "flutter_flutter", 290): ("CONTINUES-AS-BODY", "Continues with android Sdk build output."),
    ("RAGTAG", "kubernetes_kubernetes", 97): ("CONTINUES-AS-BODY", "Continues with /etc/os-release content."),
    ("RAGTAG", "kubernetes_kubernetes", 205): ("CONTINUES-AS-BODY", "Continues with kubelet event log."),
    ("RAGTAG", "kubernetes_kubernetes", 206): ("OFF-TOPIC-LOOP", "Repetitive '.' newline loop."),
    ("RAGTAG", "kubernetes_kubernetes", 211): ("MALFORMED-OUTPUT", "Just ')\\r\\n```' — no label."),
    ("RAGTAG", "kubernetes_kubernetes", 215): ("MALFORMED-OUTPUT", "'<label>...assistant\\n<label>bug</label>' — second label has correct answer."),
    ("RAGTAG", "kubernetes_kubernetes", 227): ("CHAIN-OF-THOUGHT", "Model reasons in prose ('It looks like the kubelet is failing...') instead of label."),
    ("RAGTAG", "microsoft_TypeScript", 9): ("MALFORMED-OUTPUT", "';\\r\\n```' — minimal output."),
    ("RAGTAG", "microsoft_vscode", 26): ("CONTINUES-AS-BODY", "Continues with vscode internal version strings."),
    ("RAGTAG", "microsoft_vscode", 128): ("OFF-TOPIC-LOOP", "Repetitive '8888...' digit loop."),
    ("RAGTAG", "microsoft_vscode", 179): ("OFF-TOPIC-LOOP", "Repetitive '5555...' digit loop."),
    ("RAGTAG", "microsoft_vscode", 199): ("OFF-TOPIC-LOOP", "Continues with vscode telemetry version strings."),
    ("RAGTAG", "opencv_opencv", 179): ("CONTINUES-AS-BODY", "Continues with OpenCV build configuration output."),
    ("RAGTAG", "opencv_opencv", 197): ("CONTINUES-AS-BODY", "Continues with cmake compile log + Expected behavior."),
    ("RAGTAG", "opencv_opencv", 227): ("CONTINUES-AS-BODY", "Continues with protobuf compile error."),
    ("RAGTAG", "tensorflow_tensorflow", 39): ("MALFORMED-OUTPUT", "'6\\n<label>bug</label>' — bug correct; format malformed."),
    ("RAGTAG", "tensorflow_tensorflow", 88): ("CONTINUES-AS-BODY", "Continues with binary tensor data."),
    ("RAGTAG", "tensorflow_tensorflow", 133): ("CONTINUES-AS-BODY", "Continues with hexagon delegate command line."),
    ("RAGTAG", "tensorflow_tensorflow", 172): ("MALFORMED-OUTPUT", "'el\\n```' — minimal output."),
}

expected = set(zip(df["__method"], df["__proj"], df["test_idx"]))
got = set(INV_CODES.keys())
missing = expected - got
if missing:
    print(f"WARNING: missing {len(missing)} keys")
    for k in sorted(missing): print(f"  {k}")

rows = []
for _, r in df.iterrows():
    key = (r["__method"], r["__proj"], int(r["test_idx"]))
    code, note = INV_CODES.get(key, ("OTHER", "MISSING"))
    rows.append({"method": r["__method"], "project": r["__proj"], "test_idx": int(r["test_idx"]),
                 "ground_truth": r["ground_truth"], "code": code,
                 "generated_tokens": r["generated_tokens"], "truncated": r["truncated"],
                 "note": note})
out = pd.DataFrame(rows)
out.to_csv(BASE / "codes_invalid.csv", index=False)
print(f"wrote {BASE / 'codes_invalid.csv'} ({len(out)} rows)")
print("\ncode distribution:")
print(out["code"].value_counts())
print("\ncode x method:")
print(pd.crosstab(out["code"], out["method"]))

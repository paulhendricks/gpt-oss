Hardened the compatibility harness so the behavioural fixes from runCaseProposed.ts land without the debug noise
and documented the new expectations.

- Expanded the tool-call check to tolerate items lacking rawItem, parse either string or object arguments, surface
  AJV errors, and avoid false positives before strict equality kicks in (compatibility-test/runCase.ts:143-199).
- Normalized streaming event detection to accept both wrapped and bare reasoning events, flag missing streams, and
  guard against providers that ignore the stream flag (compatibility-test/runCase.ts:113-115 and 202-255).
- Relaxed reasoning-output validation to handle missing providerData, unknown items that expose their type
  in providerData, and mixed reasoning_text/input_text segments, backed by a reusable argument normalizer
  (compatibility-test/runCase.ts:257-360).
- Updated the README to note the broader normalization behaviour and to remove the now-resolved streaming
  limitation (compatibility-test/README.md:5-76).

Validation:

- Exercised the revised helpers with targeted Node snippets to confirm the legacy crashes (missing rawItem,
  flattened streaming events, mixed reasoning segments) reproduce on the old logic and succeed with the new one.

Next steps:

1. Run npm start -- --provider <name> -n 1 once you have provider credentials to confirm end-to-end behaviour with
   a real backend.

// Regression tests for the pure arithmetic in config-calculator.html. The page is deliberately a
// single self-contained file (no build step, no CDN, works from an offline checkout), so this
// harness extracts its <script> body, runs it in a sandboxed context against a minimal DOM stub,
// and exercises the resulting top-level functions directly -- rather than restructuring the page
// into importable modules, which would give up the "one file, no tooling" property that is the
// whole point of it.
//
// Run with: node docs/config-calculator.test.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import assert from "node:assert/strict";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "config-calculator.html"), "utf8");
const scriptMatch = html.match(/<script>\n([\s\S]*?)<\/script>/);
assert.ok(scriptMatch, "could not find the <script> block in config-calculator.html");
const source = scriptMatch[1];

// --- minimal DOM stub -------------------------------------------------------------------------
// Enough for buildControls/buildDepthTable/buildSpecTable/render to run at load without throwing.
// Everything is deliberately permissive (any property can be set) rather than a faithful DOM.
function makeElement() {
  const el = {
    value: "",
    innerHTML: "",
    textContent: "",
    style: {},
    classList: { add() {}, remove() {} },
    firstChild: { nodeValue: "" },
    addEventListener() {},
  };
  // A stable child per selector, not a fresh throwaway each call, so code that queries once to
  // write (e.g. tbody.innerHTML = ...) and a test that queries again to read see the same object.
  const children = new Map();
  el.querySelector = (selector) => {
    if (!children.has(selector)) children.set(selector, makeElement());
    return children.get(selector);
  };
  return el;
}
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  },
  querySelectorAll() { return []; },
};
// A real <select> auto-selects its first <option> once populated; buildControls() only sets
// innerHTML, relying on that browser behavior to leave "model" on the first model key. Seed it
// explicitly since the stub does not replicate that; "27b" is DATA.models' first key.
document.getElementById("model").value = "27b";
document.getElementById("reserve").value = "1536";

const context = { document, console };
vm.createContext(context);
vm.runInContext(source, context, { filename: "config-calculator.html inline script" });
// Top-level `const`/`let` bindings in a vm context live in that context's script-level lexical
// environment, not as properties of the context object itself -- pull the ones under test out
// with a second expression evaluated in the same context, where they are still in scope.
const { DATA, PAGE_TOKENS, pageRoundUp, perToken, fixedBytes, decodeAtDepth, buildKvTable } =
  vm.runInContext(
    "({DATA, PAGE_TOKENS, pageRoundUp, perToken, fixedBytes, decodeAtDepth, buildKvTable})",
    context);

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log("ok   " + name);
  } catch (err) {
    failures++;
    console.log("FAIL " + name + ": " + err.message);
  }
}

// --- pageRoundUp -----------------------------------------------------------------------------
check("pageRoundUp: exact multiple stays put", () => {
  assert.equal(pageRoundUp(8192), 8192);
});
check("pageRoundUp: one token over rounds up a full page", () => {
  assert.equal(pageRoundUp(8193), 8192 + PAGE_TOKENS);
});
check("pageRoundUp: one token under the boundary still rounds up to it", () => {
  assert.equal(pageRoundUp(8191), 8192);
});
check("pageRoundUp: below one page rounds up to one page", () => {
  assert.equal(pageRoundUp(1), PAGE_TOKENS);
});

// --- decodeAtDepth -----------------------------------------------------------------------------
const model27b = DATA.models["27b"];
check("decodeAtDepth: exact at a measured depth is not floored/interpolated/extrapolated", () => {
  const d = decodeAtDepth(model27b, "int8", model27b.depths[0]);
  assert.equal(d.exact, true);
  assert.equal(d.floored, false);
  assert.equal(d.extrapolated, false);
  assert.equal(d.value, model27b.speed.int8[0]);
});
check("decodeAtDepth: below the first measured depth is floored, not interpolated", () => {
  const d = decodeAtDepth(model27b, "int8", model27b.depths[0] - 1);
  assert.equal(d.floored, true);
  assert.equal(d.exact, false);
  assert.equal(d.extrapolated, false);
  // Held at the shallowest measurement -- not some fabricated interpolated number.
  assert.equal(d.value, model27b.speed.int8[0]);
});
check("decodeAtDepth: strictly between two measured depths interpolates", () => {
  const [d0, d1] = model27b.depths;
  const mid = Math.round((d0 + d1) / 2);
  const d = decodeAtDepth(model27b, "int8", mid);
  assert.equal(d.exact, false);
  assert.equal(d.floored, undefined);
  assert.equal(d.extrapolated, false);
  const [s0, s1] = model27b.speed.int8;
  assert.ok(d.value < Math.max(s0, s1) + 1e-9 && d.value > Math.min(s0, s1) - 1e-9,
    "interpolated value should lie between the two bracketing measurements");
});
check("decodeAtDepth: past the deepest measured point extrapolates", () => {
  const deepest = model27b.depths[model27b.depths.length - 1];
  const d = decodeAtDepth(model27b, "int8", deepest * 4);
  assert.equal(d.extrapolated, true);
  assert.equal(d.exact, false);
});

// --- the PR's own verified example --------------------------------------------------------------
// From the PR description: requesting a 262,144-token INT8 context on the 27B with no speculation
// is refused by the real engine with "minimum Engine runtime reservation requires 9197389568
// bytes" -- an *incremental* reservation on top of already-resident weights, not the full device
// footprint. The page's own terms (KV + fixed sequence state + per-token remainder + workspace +
// graph allowance, excluding weights) are stated to sum to exactly that. Reproducing it here pins
// that agreement as a regression instead of a one-off manual check.
check("golden: 262144-token int8, no speculation, 27B matches the engine's own refusal figure", () => {
  const model = model27b;
  const ctx = 262144;
  const kvBytes = model.kv.int8 * pageRoundUp(ctx);
  const ovhBytes = model.seqFixedBytes + DATA.seqPerToken * ctx;
  const reservation = kvBytes + ovhBytes + model.workspaceBytes + DATA.graphBytes;
  assert.equal(reservation, 9197389568);
});

// --- max-context page alignment -----------------------------------------------------------------
check("max-context formula (mirrored here) is always a page multiple and fits", () => {
  const model = model27b;
  const pt = perToken(model, "int8");
  const fixed = fixedBytes(model, "none"); // weights + workspace + graph + seqFixedBytes, once
  for (const reserveMiB of [0, 307, 1536, 2560]) {
    const usable = (DATA.cardMiB - reserveMiB) * 1024 * 1024;
    const room = usable - fixed;
    const maxCtx = Math.max(0, PAGE_TOKENS * Math.floor(room / (PAGE_TOKENS * pt)));
    assert.equal(maxCtx % PAGE_TOKENS, 0, "reserve=" + reserveMiB + "MiB gave a non-page-aligned max");
    // Reconstructed the same way render()'s `total` is: seqFixedBytes counted once (inside
    // ovhBytes), not twice (fixedBytes() already carries its own copy for the room calculation).
    const kvBytes = model.kv.int8 * pageRoundUp(maxCtx);
    const perTokenOverhead = DATA.seqPerToken * maxCtx;
    assert.ok(fixed + kvBytes + perTokenOverhead <= usable + 1e-6,
      "reserve=" + reserveMiB + "MiB: reported max-context must not exceed usable memory");
  }
});

// buildKvTable() duplicated the same unrounded formula independently of render()'s max-context
// box; this exercises that second copy end to end, not just the shared math above.
check("buildKvTable: every format's Largest-Context figure is page-aligned", () => {
  document.getElementById("kv").value = "int8";
  document.getElementById("spec").value = "none";
  buildKvTable();
  const html = document.getElementById("kv-table").querySelector("tbody").innerHTML;
  const cells = [...html.matchAll(/<tr data-kv="(\w+)">.*?<td class="[^"]*">([\d,]+|—)( \(max\))?<\/td>/gs)];
  assert.ok(cells.length > 0, "expected at least one KV-format row to have rendered");
  for (const [, kv, numberText] of cells) {
    if (numberText === "—") continue; // an unmeasured format for this model
    const ctx = Number(numberText.replace(/,/g, ""));
    assert.equal(ctx % PAGE_TOKENS, 0, kv + "'s reported largest context is not page-aligned: " + ctx);
  }
});

console.log(failures === 0 ? "\nPASS" : "\nFAIL (" + failures + ")");
process.exit(failures === 0 ? 0 : 1);

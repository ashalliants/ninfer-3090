#!/usr/bin/env python3
"""Offline replay of expert-routing traces through the VRAM expert cache's promotion policies.

Compares (a) the policy the Flash-Next engine runs today (LFRU, modelled step for step from
src/models/qwen4_exp/program/expert_cache/expert_cache.cpp and the promotion budget in program.cpp)
with (b) a windowed-statistics policy: rank experts by routed hits in the last N rounds, re-select
every M rounds, swap only when the gain clears a hysteresis margin, at a bounded number of swaps.

    python -I tools/expert_cache_sim.py TRACE [--frames 11000] [--window 256] [--every 16] ...
    python -I tools/test_expert_cache_sim.py

THERE ARE NO REAL ROUTING TRACES IN THE REPOSITORY. Nothing here has been run on real routing data;
the tests use tiny synthetic traces whose answers were computed by hand, to prove the simulator
counts correctly, and nothing else. Do not read a policy ranking off synthetic data.

Trace format (JSON lines, optionally gzip: a name ending in .gz)
    line 1:  {"format": "ninfer-routing-trace", "version": 1, "layers": 48, "experts": 512}
    then one line per decode round, in order:
             {"t": 1, "r": [[e, e, ...], ...]}
    "t" is the tokens the round advanced (1 for plain decode, 1 + accepted drafts for an MTP round),
    "r" has one list per layer: the DISTINCT expert ids routed in that layer by all of the round's
    columns (what ExpertResidency::after_round builds as `group_`). Prefill is not part of a trace
    (the engine promotes nothing during prefill).

How to record one on the GPU box: see RECORDING below.

What is modelled (and what is not)
  * Keys are layer * experts + expert, one pool of `frames` frames shared by all layers.
  * LFRU: score = count / (now - last + 1); `now` advances once per layer per round; counts are
    never reset on eviction. Victims are the lowest (score, key) residents outside the layer-call's
    group; misses are admitted best score first, and only while they outrank the victim.
  * Budget: while fewer than `frames` promotions have happened the budget is 1 per layer per token
    (the fill phase); afterwards `due = tokens // interval` accumulates and the budget per layer is
    `due` (interval 4 by default: kDecodePromotionInterval at the reference link rate).
  * A promoted expert is usable from the next round. The engine's frame-reuse epochs, queued loads
    waiting for a free frame, the VRAM control law that resizes the pool, lent frames during
    prefill, and warm start from a saved state are NOT modelled. A hit is a routed expert resident
    at the start of its round, as ExpertResidency counts it.

RECORDING (not built or run here; no GPU or nvcc on the machine that wrote this)
  The smallest instrumentation is in ExpertResidency::after_round (expert_residency.cpp), which
  already holds each layer's distinct experts in `group_` after the dedupe loop. Behind an env var
  read once (default off), append one JSON line per round to the named file:
      if (trace_ != nullptr) {  // FILE*, opened from NINFER_ROUTE_TRACE in the constructor
          std::fprintf(trace_, "{\"t\":%d,\"r\":[", tokens);
          ... per layer: "[" + experts of group_ (key - layer * experts_) joined by "," + "]" ...
      }
  after_round gets `columns`, not tokens; pass the tokens the round advanced (settle_round has them)
  or record columns and set "t" from the request log. Write the header line once. Record a
  representative session (several thousand decode tokens of varied prompts: chat, code, long
  context), because the question is how routing drifts within and across requests.
"""
import argparse
import gzip
import json
import sys
from collections import deque

import numpy as np

FILL_BUDGET_PER_LAYER = 1  # kDecodePromotionsPerLayer
DEFAULT_INTERVAL = 4  # kDecodePromotionInterval at the reference link
DEFAULT_EXPERT_BYTES = 1.51e6  # widest record (IQ2_S), docs/maintainer/flash-next-plan.md


# --------------------------------------------------------------------------------------- traces
def _opener(path, mode):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def read_trace(path):
    """(layers, experts, rounds) with rounds a list of (tokens, [np.int64 array per layer])."""
    with _opener(path, "rt") as handle:
        header = json.loads(handle.readline())
        if header.get("format") != "ninfer-routing-trace" or header.get("version") != 1:
            raise ValueError("not a ninfer-routing-trace v1 file")
        layers, experts = int(header["layers"]), int(header["experts"])
        rounds = []
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if len(record["r"]) != layers:
                raise ValueError(f"round {len(rounds) + 1}: {len(record['r'])} layers, want {layers}")
            groups = []
            for ids in record["r"]:
                group = np.unique(np.asarray(ids, dtype=np.int64))
                if len(group) and (group[0] < 0 or group[-1] >= experts):
                    raise ValueError(f"round {len(rounds) + 1}: expert id out of range")
                groups.append(group)
            rounds.append((int(record["t"]), groups))
    return layers, experts, rounds


def write_trace(path, layers, experts, rounds):
    with _opener(path, "wt") as handle:
        handle.write(json.dumps({"format": "ninfer-routing-trace", "version": 1, "layers": layers,
                                 "experts": experts}) + "\n")
        for tokens, groups in rounds:
            handle.write(json.dumps({"t": tokens, "r": [[int(e) for e in g] for g in groups]}) + "\n")


# --------------------------------------------------------------------------------------- results
class Result:
    def __init__(self, layers):
        self.routed = np.zeros(layers, dtype=np.int64)
        self.hits = np.zeros(layers, dtype=np.int64)
        self.promotions = 0
        self.tokens = 0
        self.curve = []  # per round: cumulative (tokens, routed, hits, promotions)

    def record_round(self, tokens):
        self.tokens += tokens
        self.curve.append((self.tokens, int(self.routed.sum()), int(self.hits.sum()), self.promotions))

    @property
    def hit_rate(self):
        return float(self.hits.sum()) / max(1, int(self.routed.sum()))

    def per_layer_hit_rate(self):
        return [float(h) / max(1, int(r)) for h, r in zip(self.hits, self.routed)]

    def promotions_per_token(self):
        return self.promotions / max(1, self.tokens)

    def windowed_curve(self, buckets):
        """(tokens at slice end, hit rate, promotions/token) for each of `buckets` equal token slices."""
        if not self.curve:
            return []
        out, prev, edge = [], (0, 0, 0, 0), 0
        total = self.curve[-1][0]
        for i in range(1, buckets + 1):
            limit = total * i / buckets
            point = prev
            while edge < len(self.curve) and self.curve[edge][0] <= limit + 1e-9:
                point = self.curve[edge]
                edge += 1
            dt, dr, dh, dp = (point[0] - prev[0], point[1] - prev[1], point[2] - prev[2], point[3] - prev[3])
            out.append((point[0], dh / dr if dr else 0.0, dp / dt if dt else 0.0))
            prev = point
        return out


# --------------------------------------------------------------------------------------- policy A
class LfruSim:
    """The engine's LfruPolicy + ExpertResidency::after_round + Program::decode_budget."""

    def __init__(self, layers, experts, frames, interval=DEFAULT_INTERVAL, halving_period=0):
        self.layers, self.experts, self.frames = layers, experts, frames
        self.interval, self.halving_period = interval, halving_period
        n = layers * experts
        self.count = np.zeros(n, dtype=np.int64)
        self.last = np.zeros(n, dtype=np.int64)
        self.resident = np.zeros(n, dtype=bool)
        self.n_resident = 0
        self.now = 0
        self.promotions = 0  # ExpertResidency::Stats.promotions
        self.budget_tokens = 0
        self.last_victims = []

    def decode_budget(self, tokens):
        if self.promotions < self.frames:
            return FILL_BUDGET_PER_LAYER * tokens
        self.budget_tokens += tokens
        due = self.budget_tokens // self.interval
        self.budget_tokens %= self.interval
        return due * FILL_BUDGET_PER_LAYER

    def _scores(self, keys):
        return self.count[keys] / (self.now - self.last[keys] + 1.0)

    def _step(self, group_keys, budget):
        self.now += 1
        if self.halving_period and self.now % self.halving_period == 0:
            self.count //= 2
        self.count[group_keys] += 1
        self.last[group_keys] = self.now
        miss = group_keys[~self.resident[group_keys]]
        self.last_victims = []
        promoted = 0
        if len(miss) and budget is None:  # on-demand admission: every miss that fits (the replay tool's)
            need = self.n_resident + len(miss) - self.frames
            if need > 0:
                residents = np.flatnonzero(self.resident)
                residents = residents[~np.isin(residents, group_keys)]
                order = np.lexsort((residents, self._scores(residents)))
                for v in residents[order][:need]:
                    self.resident[v] = False
                    self.n_resident -= 1
                    self.last_victims.append(int(v))
            for k in miss:  # in the group's own order
                if self.n_resident < self.frames:
                    self.resident[k] = True
                    self.n_resident += 1
                    promoted += 1
            return promoted
        if len(miss) == 0 or budget <= 0:
            return promoted
        order = np.lexsort((miss, -self._scores(miss)))  # best score first, lower key first
        for k in miss[order]:
            if promoted >= budget:
                break
            if self.n_resident >= self.frames:
                residents = np.flatnonzero(self.resident)
                residents = residents[~np.isin(residents, group_keys)]
                if len(residents) == 0:
                    break
                scores = self._scores(residents)
                best = scores.min()
                victim = residents[scores == best].min()
                if not (self._scores(np.array([k]))[0] > best):
                    break
                self.resident[victim] = False
                self.n_resident -= 1
                self.last_victims.append(int(victim))
            self.resident[k] = True
            self.n_resident += 1
            promoted += 1
        return promoted

    def run(self, rounds):
        res = Result(self.layers)
        for tokens, groups in rounds:
            resident_at_start = self.resident.copy()
            budget = self.decode_budget(tokens)
            for layer, g in enumerate(groups):
                keys = g + layer * self.experts
                res.routed[layer] += len(keys)
                res.hits[layer] += int(resident_at_start[keys].sum())
                added = self._step(keys, budget)
                res.promotions += added
                self.promotions += added
            res.record_round(tokens)
        return res


# --------------------------------------------------------------------------------------- policy B
class WindowedSim:
    """Window hit counts over the last `window` rounds, re-selected every `every` rounds.

    Filling: the engine's fill phase (the same per-layer budget; the missed expert with the highest
    window count goes into a free frame), so cold-start differences come from the steady-state rule
    alone. Once full, at the end of every `every`-th round: pair the best non-resident candidates
    with the worst residents and swap while
        window(candidate) - window(resident) * (1 + hysteresis) >= min_gain
    holds, at most `max_swaps` swaps per re-selection (the PCIe budget). Counts are integers, so
    min_gain 1 means "strictly more hits".
    """

    def __init__(self, layers, experts, frames, window=256, every=16, hysteresis=0.0, min_gain=1,
                 max_swaps=None, interval=DEFAULT_INTERVAL):
        self.layers, self.experts, self.frames = layers, experts, frames
        self.window, self.every = window, every
        self.hysteresis, self.min_gain = hysteresis, min_gain
        # Default swap budget matches LFRU's steady-state promotion rate: `layers` per `interval` tokens.
        self.max_swaps = max_swaps if max_swaps is not None else max(1, round(layers * every / interval))
        n = layers * experts
        self.win = np.zeros(n, dtype=np.int64)
        self.ring = deque()
        self.resident = np.zeros(n, dtype=bool)
        self.n_resident = 0

    def run(self, rounds):
        res = Result(self.layers)
        for index, (tokens, groups) in enumerate(rounds, start=1):
            resident_at_start = self.resident.copy()
            all_keys = []
            for layer, g in enumerate(groups):
                keys = g + layer * self.experts
                all_keys.append(keys)
                res.routed[layer] += len(keys)
                res.hits[layer] += int(resident_at_start[keys].sum())
            round_keys = np.concatenate(all_keys) if all_keys else np.zeros(0, dtype=np.int64)
            self.win[round_keys] += 1
            self.ring.append(round_keys)
            if len(self.ring) > self.window:
                self.win[self.ring.popleft()] -= 1
            if self.n_resident < self.frames:
                res.promotions += self._fill(all_keys, tokens)
            elif index % self.every == 0:
                res.promotions += self._reselect()
            res.record_round(tokens)
        return res

    def _fill(self, all_keys, tokens):
        added = 0
        for keys in all_keys:
            miss = keys[~self.resident[keys]]
            if len(miss) == 0:
                continue
            order = np.lexsort((miss, -self.win[miss]))
            for k in miss[order][: FILL_BUDGET_PER_LAYER * tokens]:
                if self.n_resident >= self.frames:
                    return added
                self.resident[k] = True
                self.n_resident += 1
                added += 1
        return added

    def _reselect(self):
        candidates = np.flatnonzero(~self.resident & (self.win > 0))
        if len(candidates) == 0:
            return 0
        candidates = candidates[np.lexsort((candidates, -self.win[candidates]))]
        residents = np.flatnonzero(self.resident)
        residents = residents[np.lexsort((residents, self.win[residents]))]
        swaps = 0
        for c, r in zip(candidates[: self.max_swaps], residents[: self.max_swaps]):
            if self.win[c] - self.win[r] * (1.0 + self.hysteresis) < self.min_gain:
                break
            self.resident[r] = False
            self.resident[c] = True
            swaps += 1
        return swaps


# --------------------------------------------------------------------------------------- report
def summarize(name, result, expert_bytes, buckets=8):
    lines = [f"{name}: hit rate {result.hit_rate:.4f}  promotions/token {result.promotions_per_token():.3f}  "
             f"PCIe {result.promotions_per_token() * expert_bytes / 1e6:.2f} MB/token  "
             f"({result.tokens} tokens, {result.promotions} promotions)",
             "  cold-start curve [tokens: hit rate, promotions/token in the slice]: " + "  ".join(
                 f"[{t}: {h:.3f}, {p:.2f}]" for t, h, p in result.windowed_curve(buckets))]
    return "\n".join(lines)


def compare(layers, experts, rounds, frames, window, every, hysteresis, min_gain, max_swaps, interval,
            expert_bytes, per_layer):
    a = LfruSim(layers, experts, frames, interval=interval).run(rounds)
    b = WindowedSim(layers, experts, frames, window, every, hysteresis, min_gain, max_swaps, interval).run(rounds)
    print(summarize("lfru (current)", a, expert_bytes))
    print(summarize(f"windowed N={window} M={every} h={hysteresis} min_gain={min_gain}", b, expert_bytes))
    print(f"delta hit rate {b.hit_rate - a.hit_rate:+.4f}; "
          f"promotions/token {b.promotions_per_token() - a.promotions_per_token():+.3f}")
    if per_layer:
        print("per layer (layer: lfru, windowed):")
        for i, (x, y) in enumerate(zip(a.per_layer_hit_rate(), b.per_layer_hit_rate())):
            print(f"  {i:2d}: {x:.3f} {y:.3f}")
    return a, b


def main(argv=None):
    p = argparse.ArgumentParser(description="Replay a routing trace through LFRU and a windowed policy.")
    p.add_argument("trace")
    p.add_argument("--frames", type=int, default=11000, help="VRAM expert frames (docs/serving.md: ~11,000-12,500)")
    p.add_argument("--window", type=int, default=256, help="N: rounds in the statistics window")
    p.add_argument("--every", type=int, default=16, help="M: rounds between re-selections")
    p.add_argument("--hysteresis", type=float, default=0.0, help="relative margin a candidate must beat a resident by")
    p.add_argument("--min-gain", type=int, default=1, help="absolute window-count margin")
    p.add_argument("--max-swaps", type=int, default=None, help="swaps per re-selection (default: LFRU's rate)")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="LFRU promotion interval in tokens")
    p.add_argument("--expert-bytes", type=float, default=DEFAULT_EXPERT_BYTES)
    p.add_argument("--per-layer", action="store_true")
    args = p.parse_args(argv)
    layers, experts, rounds = read_trace(args.trace)
    print(f"{len(rounds)} rounds, {layers} layers, {experts} experts per layer, frames {args.frames}")
    compare(layers, experts, rounds, args.frames, args.window, args.every, args.hysteresis, args.min_gain,
            args.max_swaps, args.interval, args.expert_bytes, args.per_layer)


if __name__ == "__main__":
    sys.exit(main())

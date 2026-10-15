"""Vision smoke test through ninfer-serve: starts the server, sends image requests, checks every answer.

    python tools/vision_smoke.py --exe build-ninja/apps/ninfer-serve.exe --model models/qwen3_8_27b.ninfer \
        -- --devices 0,1 --vision --vision-residency overlay --max-concurrency 2 --max-context 24576 \
           --kv-capacity 49152 --kv-dtype int8 --prefill-chunk 512

Everything after `--` goes to ninfer-serve, so the same script covers one GPU, `--devices 0,0` (two stages on
one card), `--devices 0,1` (two cards), resident and overlay Vision. Steps: one image; two concurrent
requests with six images each (twice); one more image to show the engine still serves. Exit status is 1
if the server does not start or any request is not HTTP 200 with an answer; server errors are printed.
The images are generated (needs Pillow) with a text label in the corner, so the answers can be read.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw


def make_image(rng, tag, width=896, height=504):
    image = Image.new("RGB", (width, height), tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        x0, x1 = sorted((rng.randrange(width), rng.randrange(width)))
        y0, y1 = sorted((rng.randrange(height), rng.randrange(height)))
        draw.rectangle([x0, y0, x1, y1], fill=tuple(rng.randrange(256) for _ in range(3)))
    draw.text((20, 20), f"IMAGE {tag}", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, help="ninfer-serve executable")
    parser.add_argument("--model", required=True, help="a vision-capable .ninfer artifact")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--start-timeout", type=int, default=900, help="seconds to wait for the server")
    parser.add_argument("--images", default="6", help="images per request in the concurrent steps; two comma-separated counts "
                        "(for example 10,9) give the two requests different numbers")
    parser.add_argument("--width", type=int, default=896, help="image width in pixels")
    parser.add_argument("--height", type=int, default=504, help="image height in pixels")
    parser.add_argument("--rounds", type=int, default=2, help="rounds of two concurrent requests")
    parser.add_argument("--random-sizes", action="store_true",
                        help="give every image a random size between 56x56 and --width x --height, so image token "
                             "counts and chunk boundaries vary")
    parser.add_argument("--seed", type=int, default=7, help="seed of the generated images")
    parser.add_argument("--shared-images", action="store_true",
                        help="build image i of every request from the same seed, so concurrent requests and later rounds "
                             "carry identical (cacheable) images")
    parser.add_argument("--effort", default="none", help="reasoning_effort sent with each request")
    parser.add_argument("--max-tokens", type=int, default=40, help="output tokens per request")
    parser.add_argument("serve_flags", nargs="*", help="after --: flags for ninfer-serve")
    args = parser.parse_args()
    args.exe, args.model = os.path.abspath(args.exe), os.path.abspath(args.model)

    base = f"http://127.0.0.1:{args.port}"
    log_file = tempfile.NamedTemporaryFile("w+", suffix=".log", prefix="vision-smoke-", delete=False)
    flags = list(args.serve_flags)
    if "--vision" not in flags:
        flags.append("--vision")
    creation = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    server = subprocess.Popen([args.exe, args.model, "--host", "127.0.0.1", "--port", str(args.port)] + flags,
                              stdout=log_file, stderr=subprocess.STDOUT, creationflags=creation)
    rng = random.Random(args.seed)
    failures = []
    model_id = None

    def pick_size(source):
        if not args.random_sizes:
            return args.width, args.height
        return source.randrange(56, args.width + 1), source.randrange(56, args.height + 1)

    def ask(images, tag, results, key):
        content = [{"type": "text", "text": "Briefly: what is the text written in the top-left corner of each image?"}]
        # A shared image takes pixels and size from its own seed, so every request sees identical bytes.
        sources = [random.Random(1000 + i) if args.shared_images else rng for i in range(images)]
        content += [{"type": "image_url",
                     "image_url": {"url": make_image(sources[i],
                                                     f"s-{i}" if args.shared_images else f"{tag}-{i}",
                                                     *pick_size(sources[i]))}}
                    for i in range(images)]
        body = {"model": model_id, "messages": [{"role": "user", "content": content}], "max_tokens": args.max_tokens,
                "temperature": 0, "reasoning_effort": args.effort}
        request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read())
            message = data["choices"][0]["message"]
            text = (message.get("content") or "").strip()
            thought = (message.get("reasoning_content") or "").strip()
            generated = data["usage"].get("completion_tokens", 0)
            shown = text or (("(thinking) " + thought) if thought else f"({generated} tokens, all hidden thinking)")
            results[key] = ("200", data["usage"]["prompt_tokens"], shown[:70].replace("\n", " "), generated > 0)
        except urllib.error.HTTPError as error:
            results[key] = (str(error.code), 0, error.read()[:160].decode(errors="replace"), False)
        except Exception as error:  # noqa: BLE001
            results[key] = ("ERR", 0, str(error)[:160], False)

    def report(step, results):
        for key, (status, tokens, text, ok) in sorted(results.items()):
            print(f"  {step} [{key}]: HTTP {status}, {tokens} prompt tokens, {text!r}")
            if not ok:
                failures.append(f"{step} [{key}]: HTTP {status} {text}")

    try:
        started = time.time()
        while True:
            if server.poll() is not None:
                print(f"FAIL: the server exited at startup with code {server.returncode}")
                failures.append("server did not start")
                break
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
            if time.time() - started > args.start_timeout:
                print("FAIL: the server did not become healthy in time")
                failures.append("server start timeout")
                break
        if not failures:
            model_id = json.loads(urllib.request.urlopen(base + "/v1/models").read())["data"][0]["id"]
            print(f"server up ({model_id}) with: {' '.join(flags)}")
            results = {}
            ask(1, "single", results, "image")
            report("one image", results)
            counts = [int(n) for n in args.images.split(",")]
            counts = (counts * 2)[:2]
            for round_number in range(1, args.rounds + 1):
                results = {}
                threads = [threading.Thread(target=ask, args=(counts[k], f"c{round_number}{k}", results, f"request {k}"))
                           for k in (0, 1)]
                [t.start() for t in threads]
                [t.join() for t in threads]
                report(f"two concurrent requests, {counts[0]} and {counts[1]} images, round {round_number}", results)
            results = {}
            ask(1, "after", results, "image")
            report("one image after", results)
    finally:
        if server.poll() is None:
            server.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
        log_file.flush()
        log_file.seek(0)
        errors = [line.rstrip() for line in log_file if re.search(r"\bERROR\b|worker failure|Exception", line)]
        if errors:
            print("server errors:")
            for line in errors[:6]:
                print("  " + line[:220])
            failures.append("the server logged errors")
        print(f"server log: {log_file.name}")
    print("PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

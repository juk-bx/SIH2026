"""
asr_server_reference.py — minimal reference server for the cloud side of
the pipeline. Not part of the graded edge deliverable, but useful for
demoing "hybrid architecture: edge wakes up, cloud does the heavy lifting"
end-to-end during evaluation.

Protocol (see esp32/naad_kws/main/network_stream.c):
    Each utterance is sent as:
        uint32 magic        (0x4441414E, little-endian)
        uint32 num_samples
        int16[num_samples]  raw PCM, 16kHz mono

Run:
    python asr_server_reference.py --port 5005

Swap `run_asr()` for a real call to any open-source ASR engine you like —
e.g. faster-whisper, whisper.cpp, or Vosk — none of which are excluded by
the challenge's software restrictions (those target the *wake-word*
detector, not the downstream ASR).
"""

import argparse
import socket
import struct
import time
import wave
from pathlib import Path

MAGIC = 0x4441414E
HEADER_FMT = "<II"
HEADER_LEN = struct.calcsize(HEADER_FMT)
SAMPLE_RATE = 16000


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def run_asr(pcm_bytes, utterance_id):
    """
    Placeholder "cloud does the heavy lifting" step. Replace this with a
    real ASR call. Left as a stub + wav dump so you can sanity-check the
    edge->cloud audio path is intact even before wiring up a real engine.
    """
    out_dir = Path("received_utterances")
    out_dir.mkdir(exist_ok=True)
    wav_path = out_dir / f"utterance_{utterance_id:04d}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    print(f"  [asr] saved {wav_path} ({len(pcm_bytes) / 2 / SAMPLE_RATE:.2f}s) "
          f"-> hand this to your ASR engine of choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Listening on {args.host}:{args.port} for the ESP32 keep-alive "
          f"connection...")

    utterance_id = 0
    while True:
        conn, addr = server.accept()
        print(f"ESP32 connected from {addr}")
        try:
            while True:
                header = recv_exact(conn, HEADER_LEN)
                if header is None:
                    print("ESP32 disconnected")
                    break
                magic, num_samples = struct.unpack(HEADER_FMT, header)
                if magic != MAGIC:
                    print(f"  [warn] bad magic 0x{magic:08x}, dropping connection")
                    break

                recv_start = time.time()
                pcm_bytes = recv_exact(conn, num_samples * 2)
                if pcm_bytes is None:
                    break
                recv_ms = (time.time() - recv_start) * 1000
                print(f"Received utterance: {num_samples} samples "
                      f"({num_samples / SAMPLE_RATE:.2f}s) in {recv_ms:.1f} ms")

                utterance_id += 1
                run_asr(pcm_bytes, utterance_id)
        finally:
            conn.close()


if __name__ == "__main__":
    main()

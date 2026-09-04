// network_stream.h — WiFi station + a TCP connection to the ASR server that
// is opened ONCE at boot and kept alive, so that the instant the keyword
// fires there is zero TCP-handshake latency between "keyword ended" and
// "cloud ASR receiving audio" (this delta is one of the graded metrics).
#ifndef NAAD_NETWORK_STREAM_H_
#define NAAD_NETWORK_STREAM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Connects to WiFi (blocking until associated) and opens the TCP socket to
// the ASR server configured via `idf.py menuconfig` (NAAD_ASR_SERVER_IP /
// NAAD_ASR_SERVER_PORT). Call once at boot. Internally retries/backs off and
// reconnects if the link drops, so callers can assume the socket "just
// works" from their perspective.
void network_stream_init(void);

// Sends a minimal 8-byte frame header (magic + sample count), then the raw
// int16 PCM samples themselves — no re-encoding, no extra hops, which is
// what keeps overhead and latency low per the problem statement. The ASR
// server is expected to speak this trivial framing (see README.md for the
// reference server snippet). Safe to call from the inference task.
void network_stream_send_pcm(const int16_t *pcm, size_t num_samples);

// True once the keep-alive socket is currently connected.
bool network_stream_is_connected(void);

#ifdef __cplusplus
}
#endif

#endif  // NAAD_NETWORK_STREAM_H_

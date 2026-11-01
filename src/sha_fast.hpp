// sha_fast.hpp — SHA-256 for the timed extract phase (2026-11-01, v5 pipeline work).
//
// Uses the platform SHA implementation (OpenSSL's one-shot SHA256, which dispatches to the
// SHA-NI instructions on this CPU) and falls back to the shipped software implementation
// (sha256.hpp) when BWH_SHA=sw is selected, so both native and baseline arms can be compared
// with the identical extract path.  New file; the shipped sha256.hpp is untouched.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <openssl/sha.h>

#include "sha256.hpp"

namespace bwh {

// The extract phase both engines charge: one complete-vector checksum + SHA-256.
inline bool sha_use_hw() {
  const char* e = std::getenv("BWH_SHA");
  if (e && std::strcmp(e, "sw") == 0) return false;
  return true;  // SHA-NI via OpenSSL by default; BWH_SHA=sw reproduces the software path
}

inline std::string answers_sha256(const void* data, size_t n) {
  unsigned char out[32];
  if (sha_use_hw()) {
    SHA256(static_cast<const unsigned char*>(data), n, out);
  } else {
    const std::string sw = sha::of(data, n);
    return sw;
  }
  static const char* hex = "0123456789abcdef";
  std::string s;
  s.resize(64);
  for (size_t i = 0; i < 32; ++i) {
    s[2 * i] = hex[out[i] >> 4];
    s[2 * i + 1] = hex[out[i] & 15];
  }
  return s;
}

}  // namespace bwh

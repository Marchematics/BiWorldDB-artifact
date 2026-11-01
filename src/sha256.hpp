// SHA-256 (FIPS 180-4), self-contained; shared by the worker answer digest.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

namespace bwh {
namespace sha {

struct Ctx {
  u32 h[8];
  u64 total;
  u8 buf[64];
  size_t fill;
};

static const u32 K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static inline u32 rotr(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }

static void transform(Ctx& c, const u8* p) {
  u32 w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (u32(p[4 * i]) << 24) | (u32(p[4 * i + 1]) << 16) |
           (u32(p[4 * i + 2]) << 8) | u32(p[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    u32 s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    u32 s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  u32 a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3];
  u32 e = c.h[4], f = c.h[5], g = c.h[6], hh = c.h[7];
  for (int i = 0; i < 64; ++i) {
    u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    u32 ch = (e & f) ^ (~e & g);
    u32 t1 = hh + S1 + ch + K[i] + w[i];
    u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    u32 maj = (a & b) ^ (a & cc) ^ (b & cc);
    u32 t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = cc; cc = b; b = a; a = t1 + t2;
  }
  c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d;
  c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += hh;
}

static void init(Ctx& c) {
  c.h[0] = 0x6a09e667u; c.h[1] = 0xbb67ae85u; c.h[2] = 0x3c6ef372u;
  c.h[3] = 0xa54ff53au; c.h[4] = 0x510e527fu; c.h[5] = 0x9b05688cu;
  c.h[6] = 0x1f83d9abu; c.h[7] = 0x5be0cd19u;
  c.total = 0;
  c.fill = 0;
}

static void update(Ctx& c, const void* data, size_t n) {
  const u8* p = static_cast<const u8*>(data);
  c.total += u64(n);
  if (c.fill) {
    size_t take = 64 - c.fill;
    if (take > n) take = n;
    std::memcpy(c.buf + c.fill, p, take);
    c.fill += take;
    p += take;
    n -= take;
    if (c.fill == 64) {
      transform(c, c.buf);
      c.fill = 0;
    }
  }
  while (n >= 64) {
    transform(c, p);
    p += 64;
    n -= 64;
  }
  if (n) {
    std::memcpy(c.buf, p, n);
    c.fill = n;
  }
}

static void finish(Ctx& c, u8 out[32]) {
  const u64 bits = c.total * 8;
  u8 pad[72];
  size_t padlen = (c.fill < 56) ? (56 - c.fill) : (120 - c.fill);
  pad[0] = 0x80;
  std::memset(pad + 1, 0, padlen - 1);
  for (int i = 0; i < 8; ++i) pad[padlen + i] = u8(bits >> (56 - 8 * i));
  update(c, pad, padlen + 8);
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = u8(c.h[i] >> 24);
    out[4 * i + 1] = u8(c.h[i] >> 16);
    out[4 * i + 2] = u8(c.h[i] >> 8);
    out[4 * i + 3] = u8(c.h[i]);
  }
}

static std::string hex(const u8 d[32]) {
  static const char* hx = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 32; ++i) {
    s[size_t(2 * i)] = hx[d[i] >> 4];
    s[size_t(2 * i + 1)] = hx[d[i] & 15];
  }
  return s;
}

static std::string of(const void* data, size_t n) {
  Ctx c;
  init(c);
  update(c, data, n);
  u8 d[32];
  finish(c, d);
  return hex(d);
}

}  // namespace sha
}  // namespace bwh

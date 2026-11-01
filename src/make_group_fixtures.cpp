// make_group_fixtures.cpp — derive reduced-group fixtures for the cross-version region sweep.
// New file (2026-11-02).  Reads a BWBASE02 fixture and writes variants whose group ids are
// remapped g -> g % Gnew, with base factors/labels taken from the first Gnew original groups.
// Row set, timestamps and amounts are unchanged; only the group dimension is collapsed, so the
// same derived fixture can be fed to every compared arm identically.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common.hpp"

using namespace bwh;

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: %s in.bin Gnew out.bin\n", argv[0]); return 2; }
  Fixture f;
  if (!load_fixture(argv[1], f)) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 3; }
  const i32 Gnew = std::atoi(argv[2]);
  if (Gnew < 1 || Gnew > f.h.groups) { std::fprintf(stderr, "bad Gnew\n"); return 3; }
  FixtureHeader h = f.h;
  h.groups = Gnew;
  FILE* out = std::fopen(argv[3], "wb");
  if (!out) { std::fprintf(stderr, "cannot write\n"); return 4; }
  if (std::fwrite(&h, sizeof(h), 1, out) != 1) return 4;
  for (i64 i = 0; i < h.n; ++i) {
    BaseRec r = f.rows[size_t(i)];
    r.g = r.g % Gnew;
    if (std::fwrite(&r, sizeof(r), 1, out) != 1) return 4;
  }
  for (i32 g = 0; g < Gnew; ++g) {
    const i64 fac = f.base_factor[size_t(g)];
    if (fac <= 0) { std::fprintf(stderr, "bad factor\n"); return 4; }
    if (std::fwrite(&fac, sizeof(fac), 1, out) != 1) return 4;
  }
  for (i32 g = 0; g < Gnew; ++g) {
    const i32 lab = f.group_labels[size_t(g)];
    if (std::fwrite(&lab, sizeof(lab), 1, out) != 1) return 4;
  }
  std::fclose(out);
  std::printf("wrote %s rows=%lld groups=%d\n", argv[3], (long long)h.n, Gnew);
  return 0;
}

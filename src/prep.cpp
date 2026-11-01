// prep.cpp — one-time preparation of a public base into a binary fixture.
//
// Two admitted bases are supported:
//   --format intel   : gzip-decompressed Intel Lab Data text on stdin
//                      (columns: date time epoch moteid temperature humidity light voltage)
//   --format uci501  : one or more UCI Beijing multi-site CSV files (PM2.5, 12 stations)
//
// Admitted rows follow the v7 contract: non-NULL group/time/numeric predicate, no
// zero imputation. Timestamps and amounts are exact integers:
//   intel : signed-64 naive-civil microseconds since 2004-01-01; amount = micro C
//   uci501: civil-hour index since 2013-03-01 00:00;                 amount = micro g/m3
// Both use base multiplier 1000 (thousandths) and amount scale 1e6.
#include <cerrno>
#include <cinttypes>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"

using namespace bwh;

static i64 days_from_civil(i64 y, unsigned m, unsigned d) {
  y -= m <= 2;
  const i64 era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + i64(doe) - 719468;
}

static bool parse_date(const char* s, int& y, int& mo, int& d) {
  return std::sscanf(s, "%d-%d-%d", &y, &mo, &d) == 3;
}

static bool parse_time_us(const char* s, i64& out_us) {
  int h = 0, m = 0, sec = 0, n = 0;
  if (std::sscanf(s, "%d:%d:%d%n", &h, &m, &sec, &n) != 3) return false;
  const char* p = s + n;
  i64 us = 0;
  if (*p == '.') {
    ++p;
    i64 scale = 100000;
    int fd = 0;
    bool any = false;
    while (*p >= '0' && *p <= '9') {
      if (fd >= 6) return false;  // reject non-exact input
      us += (*p - '0') * scale;
      scale /= 10;
      ++fd;
      any = true;
      ++p;
    }
    if (!any || *p != '\0') return false;
  } else if (*p != '\0') {
    return false;
  }
  out_us = (i64(h) * 3600 + i64(m) * 60 + i64(sec)) * 1000000 + us;
  return true;
}

// Exact decimal -> integer scaled by 1e6. Rejects more than 6 fractional digits.
static bool parse_scaled6(const std::string& s, i64& out) {
  if (s.empty()) return false;
  const char* p = s.c_str();
  bool neg = false;
  if (*p == '-') { neg = true; ++p; }
  else if (*p == '+') ++p;
  i64 ip = 0;
  while (*p >= '0' && *p <= '9') { ip = ip * 10 + (*p - '0'); ++p; }
  i64 frac = 0;
  int fd = 0;
  if (*p == '.') {
    ++p;
    while (*p >= '0' && *p <= '9') {
      if (fd >= 6) return false;
      frac = frac * 10 + (*p - '0');
      ++fd;
      ++p;
    }
  }
  if (*p != '\0') return false;
  while (fd < 6) { frac *= 10; ++fd; }
  out = ip * 1000000 + frac;
  if (neg) out = -out;
  return true;
}

// The Intel source separates columns by single spaces and represents a missing
// field as an empty column (consecutive spaces). Splitting on each single space
// therefore recovers the exact column positions; collapsing whitespace would
// shift later columns and silently admit rows whose temperature is NULL.
static void split_ws(const std::string& line, std::vector<std::string>& tok) {
  tok.clear();
  size_t start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ' ') {
      tok.emplace_back(line.substr(start, i - start));
      start = i + 1;
    }
  }
  // Drop trailing empty columns produced by trailing spaces / \r.
  while (!tok.empty() && tok.back().empty()) tok.pop_back();
}

struct Builder {
  std::vector<i64> ts, amt;
  std::vector<int> grp;  // dense
  std::vector<i64> labels;
  std::unordered_map<i64, int> label_map;
  i64 lines = 0, missing = 0, numeric = 0;
  i64 tmin = INT64_MAX, tmax = INT64_MIN;

  int group_of(i64 label) {
    auto it = label_map.find(label);
    if (it != label_map.end()) return it->second;
    int g = int(labels.size());
    label_map.emplace(label, g);
    labels.push_back(label);
    return g;
  }
  void add(i64 t, i32 g, i64 a) {
    ts.push_back(t);
    amt.push_back(a);
    grp.push_back(g);
    if (t < tmin) tmin = t;
    if (t > tmax) tmax = t;
    ++numeric;
  }
};

static bool run_intel(Builder& b, i64 sample, i32& unit, i64& amount_scale) {
  unit = 0;
  amount_scale = 6;
  std::vector<std::string> tok;
  std::string line;
  line.reserve(256);
  const i64 base_days = days_from_civil(2004, 1, 1);
  while (std::getline(std::cin, line)) {
    ++b.lines;
    if (line.empty()) continue;
    split_ws(line, tok);
    if (tok.size() < 5) { ++b.missing; continue; }
    int y = 0, mo = 0, d = 0;
    i64 tod = 0;
    if (tok[0].empty() || tok[1].empty() || !parse_date(tok[0].c_str(), y, mo, d) ||
        !parse_time_us(tok[1].c_str(), tod)) { ++b.missing; continue; }
    int mote = 0;
    if (tok[3].empty() || std::sscanf(tok[3].c_str(), "%d", &mote) != 1) { ++b.missing; continue; }
    // Retain every observed source group label (the dimension S has one multiplier
    // for every observed label, even if that label has no admitted numeric row).
    const int dense_group = b.group_of(mote);
    i64 a = 0;
    if (tok[4].empty() || !parse_scaled6(tok[4], a)) { ++b.missing; continue; }
    i64 t = (days_from_civil(i64(y), unsigned(mo), unsigned(d)) - base_days) * 86400000000LL + tod;
    b.add(t, dense_group, a);
    if (sample > 0 && b.numeric >= sample) break;
  }
  return b.numeric > 0;
}

// Split a CSV row on commas (the UCI files contain no quoted commas in the fields we
// read; wind direction may be empty which stays an empty field).
static std::string strip_quotes(std::string s) {
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
  return s;
}

static void split_csv(const std::string& line, std::vector<std::string>& out) {
  out.clear();
  size_t start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      out.emplace_back(strip_quotes(line.substr(start, i - start)));
      start = i + 1;
    }
  }
}

static bool run_uci(Builder& b, const std::vector<std::string>& paths, i32& unit, i64& amount_scale) {
  unit = 1;
  amount_scale = 6;
  const i64 base_days = days_from_civil(2013, 3, 1);
  std::vector<std::string> fields;
  std::string line;
  for (const auto& path : paths) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { std::fprintf(stderr, "cannot open %s: %s\n", path.c_str(), std::strerror(errno)); return false; }
    char buf[4096];
    bool first = true;
    while (std::fgets(buf, sizeof(buf), fp)) {
      line.assign(buf);
      while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
      if (line.empty()) continue;
      if (first && (line.rfind("No,", 0) == 0 || line.rfind("\"No\"", 0) == 0)) { first = false; continue; }
      first = false;
      ++b.lines;
      split_csv(line, fields);
      if (fields.size() < 18) { ++b.missing; continue; }
      int y = 0, mo = 0, d = 0, hh = 0, no = 0;
      if (std::sscanf(fields[0].c_str(), "%d", &no) != 1 ||
          std::sscanf(fields[1].c_str(), "%d", &y) != 1 ||
          std::sscanf(fields[2].c_str(), "%d", &mo) != 1 ||
          std::sscanf(fields[3].c_str(), "%d", &d) != 1 ||
          std::sscanf(fields[4].c_str(), "%d", &hh) != 1) {
        ++b.missing;
        continue;
      }
      i64 a = 0;
      if (!parse_scaled6(fields[5], a)) { ++b.missing; continue; }  // PM2.5 NULL -> excluded
      const std::string& station = fields[17];
      i64 label = 0;
      for (unsigned char c : station) label = label * 131 + c;  // stable station ordinal
      i64 t = (days_from_civil(i64(y), unsigned(mo), unsigned(d)) - base_days) * 24 + hh;
      b.add(t, b.group_of(label), a);
    }
    std::fclose(fp);
  }
  return b.numeric > 0;
}

int main(int argc, char** argv) {
  std::string out_path, format = "intel";
  i64 sample = -1;
  std::vector<std::string> inputs;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
    else if (!std::strcmp(argv[i], "--format") && i + 1 < argc) format = argv[++i];
    else if (!std::strcmp(argv[i], "--sample") && i + 1 < argc) sample = std::atoll(argv[++i]);
    else inputs.push_back(argv[i]);
  }
  if (out_path.empty()) {
    std::fprintf(stderr, "usage: prep --format intel|uci501 --out FIXTURE.bin [--sample N] [inputs]\n");
    return 2;
  }
  Builder b;
  i32 unit = 0;
  i64 amount_scale = 6;
  bool ok = (format == "intel") ? run_intel(b, sample, unit, amount_scale)
                                : run_uci(b, inputs, unit, amount_scale);
  if (!ok) { std::fprintf(stderr, "no admitted rows parsed\n"); return 3; }

  const i32 G = i32(b.labels.size());
  // Deterministic dense group mapping: sort labels ascending.
  std::vector<int> order(static_cast<size_t>(G));
  for (int g = 0; g < G; ++g) order[size_t(g)] = g;
  std::sort(order.begin(), order.end(), [&](int x, int y) { return b.labels[size_t(x)] < b.labels[size_t(y)]; });
  std::vector<int> remap(static_cast<size_t>(G));
  std::vector<i32> sorted_labels(static_cast<size_t>(G));
  for (int k = 0; k < G; ++k) {
    remap[size_t(order[size_t(k)])] = k;
    sorted_labels[size_t(k)] = i32(order[size_t(k)]);
  }

  FixtureHeader h{};
  std::memcpy(h.magic, "BWBASE02", 8);
  h.n = b.numeric;
  h.groups = G;
  h.unit = unit;
  h.original_lines = b.lines;
  h.missing_values = b.missing;
  h.t_min = b.tmin;
  h.t_max = b.tmax;
  h.amount_scale = amount_scale;

  std::vector<BaseRec> recs(size_t(b.numeric));
  for (i64 i = 0; i < b.numeric; ++i) {
    recs[size_t(i)].t = b.ts[size_t(i)];
    recs[size_t(i)].amount = b.amt[size_t(i)];
    recs[size_t(i)].g = remap[size_t(b.grp[size_t(i)])];
    recs[size_t(i)].pad = 0;
  }
  std::vector<i64> factors(size_t(G), 1000);

  FILE* fp = std::fopen(out_path.c_str(), "wb");
  if (!fp) { std::fprintf(stderr, "cannot write %s: %s\n", out_path.c_str(), std::strerror(errno)); return 4; }
  bool wok = std::fwrite(&h, sizeof(h), 1, fp) == 1 &&
             std::fwrite(recs.data(), sizeof(BaseRec), recs.size(), fp) == recs.size() &&
             std::fwrite(factors.data(), sizeof(i64), factors.size(), fp) == factors.size() &&
             std::fwrite(sorted_labels.data(), sizeof(i32), sorted_labels.size(), fp) == sorted_labels.size();
  std::fclose(fp);
  if (!wok) { std::fprintf(stderr, "short write\n"); return 5; }

  std::string meta = out_path + ".meta.json";
  fp = std::fopen(meta.c_str(), "w");
  if (fp) {
    std::fprintf(fp, "{\n  \"format\": \"%s\",\n  \"source_lines\": %" PRId64 ",\n  \"base_rows\": %" PRId64
                     ",\n  \"excluded_null_numeric_rows\": %" PRId64 ",\n  \"groups\": %d,\n",
                 format.c_str(), b.lines, b.numeric, b.missing, G);
    std::fprintf(fp, "  \"timestamp_unit\": \"%s\",\n  \"t_min\": %" PRId64 ",\n  \"t_max\": %" PRId64
                     ",\n  \"timestamp_span\": %" PRId64 ",\n  \"amount_scale\": %" PRId64 ",\n",
                 unit == 0 ? "naive-civil microsecond since 2004-01-01" : "civil-hour index since 2013-03-01 00:00",
                 b.tmin, b.tmax, b.tmax - b.tmin + 1, amount_scale);
    std::fprintf(fp, "  \"base_multiplier_thousandths\": 1000\n}\n");
    std::fclose(fp);
  }
  std::printf("format=%s base_rows=%" PRId64 " excluded=%" PRId64 " groups=%d t_min=%" PRId64 " t_max=%" PRId64 "\n",
              format.c_str(), b.numeric, b.missing, G, b.tmin, b.tmax);
  return 0;
}

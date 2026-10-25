/**
 * @file scoring.cpp
 * @brief ScoreModifiers (recency decay, multipliers, boosts) and dates.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "filter.h"
#include "metadata.h"

namespace vecdb {

namespace {

std::optional<double> numeric(const Metadata& md, const std::string& field) {
    const std::optional<Value> v = md.get(field);
    if (!v) return std::nullopt;
    if (v->type == FieldType::Int) return double(v->i);
    if (v->type == FieldType::Float) return v->f;
    if (v->type == FieldType::Bool) return v->b ? 1.0 : 0.0;
    return std::nullopt;  // keywords and tags are not numbers: treated as missing
}

bool finite(double x) { return std::isfinite(x); }

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ScoreModifiers: ") + what);
}

}  // namespace

void ScoreModifiers::validate() const {
    for (const RecencyDecay& d : decays) {
        need(!d.field.empty(), "a decay needs a field");
        need(finite(d.origin), "a decay's origin must be finite");
        need(finite(d.half_life) && d.half_life > 0, "a decay's half_life must be positive and finite");
        need(finite(d.missing) && d.missing >= 0, "a decay's missing factor must be finite and non-negative");
        need(d.floor >= 0 && d.floor <= 1, "a decay's floor must be in [0, 1]");
        need(d.shape == DecayShape::Exponential || d.shape == DecayShape::Linear, "unknown decay shape");
    }
    for (const FieldMultiplier& m : multipliers) {
        need(!m.field.empty(), "a multiplier needs a field");
        need(finite(m.missing) && m.missing >= 0, "a multiplier's missing factor must be finite and non-negative");
        need(finite(m.min) && m.min >= 0, "a multiplier's min must be finite and non-negative");
        need(!(m.max < m.min) && !std::isnan(m.max), "a multiplier's max must not be below its min");
    }
    for (const MatchBoost& b : boosts) need(finite(b.bonus), "a boost's bonus must be finite");
}

// GCC 16 reports -Wstringop-overflow and -Warray-bounds inside std::vector::resize once
// MetadataStore::write is inlined here ("8 bytes into a region of size 1"): a false positive of its
// flow analysis on plain vector growth, which runs clean under AddressSanitizer. Silenced for this
// function only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
double ScoreModifiers::apply(double base, const Metadata& md) const {
    double factor = 1.0;
    for (const RecencyDecay& d : decays) {
        const std::optional<double> v = numeric(md, d.field);
        double f = d.missing;
        if (v) {
            const double dt = std::max(0.0, d.origin - *v);
            f = d.shape == DecayShape::Exponential ? std::exp2(-dt / d.half_life) : std::max(0.0, 1.0 - dt / (2.0 * d.half_life));
            f = std::max(d.floor, f);
        }
        factor *= f;
    }
    for (const FieldMultiplier& m : multipliers) {
        const std::optional<double> v = numeric(md, m.field);
        factor *= v ? std::clamp(*v, m.min, m.max) : m.missing;
    }
    double score = base * factor;
    if (!boosts.empty()) {
        // The record alone in a store of its own: the filter's exact rules (types,
        // missing fields, tags) apply, as in a filtered search.
        MetadataStore one;
        (void)one.write(0, md);
        for (const MatchBoost& b : boosts)
            if (CompiledFilter(b.when, one).matches(0)) score += b.bonus;
    }
    return score;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// ---------------------------------------------------------------- dates

namespace {

// Days since 1970-01-01 of a proleptic Gregorian date (H. Hinnant's algorithm).
std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

void civil_from_days(std::int64_t z, std::int64_t& y, unsigned& m, unsigned& d) {
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<std::int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp < 10 ? mp + 3 : mp - 9;
    y += m <= 2;
}

bool leap(std::int64_t y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

}  // namespace

std::int64_t parse_date(std::string_view s) {
    const std::string text(s);
    auto bad = [&]() -> std::int64_t { throw std::invalid_argument("parse_date: not an ISO 8601 date: \"" + text + "\""); };
    std::size_t i = 0;
    auto digits = [&](std::size_t n) -> long long {
        if (i + n > s.size()) bad();
        long long v = 0;
        for (std::size_t j = 0; j < n; ++j) {
            const char c = s[i + j];
            if (c < '0' || c > '9') bad();
            v = v * 10 + (c - '0');
        }
        i += n;
        return v;
    };
    auto expect = [&](char c) { if (i >= s.size() || s[i] != c) bad(); ++i; };
    const long long year = digits(4);
    expect('-');
    const long long month = digits(2);
    expect('-');
    const long long day = digits(2);
    static const unsigned mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12) bad();
    const unsigned dim = mdays[month - 1] + (month == 2 && leap(year) ? 1 : 0);
    if (day < 1 || day > dim) bad();
    long long hh = 0, mm = 0, ss = 0, ms = 0, offset = 0;
    if (i < s.size() && (s[i] == 'T' || s[i] == ' ')) {
        ++i;
        hh = digits(2);
        expect(':');
        mm = digits(2);
        if (i < s.size() && s[i] == ':') {
            ++i;
            ss = digits(2);
            if (i < s.size() && s[i] == '.') {
                ++i;
                std::size_t n = 0;
                long long frac = 0;
                while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                    if (n < 3) frac = frac * 10 + (s[i] - '0');
                    ++n;
                    ++i;
                }
                if (n == 0) bad();
                for (; n < 3; ++n) frac *= 10;
                ms = frac;
            }
        }
        if (hh > 23 || mm > 59 || ss > 59) bad();
        if (i < s.size() && s[i] == 'Z') {
            ++i;
        } else if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            const int sign = s[i] == '-' ? -1 : 1;
            ++i;
            const long long oh = digits(2);
            expect(':');
            const long long om = digits(2);
            if (oh > 23 || om > 59) bad();
            offset = sign * (oh * 60 + om);
        }
    }
    if (i != s.size()) bad();
    const std::int64_t days = days_from_civil(year, unsigned(month), unsigned(day));
    return ((days * 24 + hh) * 60 + mm - offset) * 60000 + ss * 1000 + ms;
}

std::string format_date(std::int64_t ms) {
    std::int64_t days = ms >= 0 ? ms / 86400000 : -((-ms + 86399999) / 86400000);
    std::int64_t rest = ms - days * 86400000;
    std::int64_t y;
    unsigned m, d;
    civil_from_days(days, y, m, d);
    char buf[96];  // room for any 64-bit year
    std::snprintf(buf, sizeof buf, "%04lld-%02u-%02uT%02lld:%02lld:%02lld.%03lldZ", static_cast<long long>(y), m, d,
                  static_cast<long long>(rest / 3600000), static_cast<long long>(rest / 60000 % 60), static_cast<long long>(rest / 1000 % 60),
                  static_cast<long long>(rest % 1000));
    return buf;
}

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace vecdb
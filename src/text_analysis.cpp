#include "text_analysis.h"

#include <algorithm>
#include <array>
#include <utility>

namespace vecdb {

namespace {

bool vowel(char c) { return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y'; }

// The last letter first: most suffixes fail there, without a full comparison.
bool ends(const std::string& w, std::string_view s) {
    return w.size() >= s.size() && w.back() == s.back() && std::string_view(w).substr(w.size() - s.size()) == s;
}

bool has_vowel(const std::string& w, std::size_t end) {
    for (std::size_t i = 0; i < end; ++i)
        if (vowel(w[i])) return true;
    return false;
}

/// A short syllable ending at `end`: consonant, vowel, consonant (not w, x or Y); a vowel
/// then a consonant at the start; or "past".
bool short_syllable(const std::string& w, std::size_t end) {
    if (end >= 3) {
        const char c = w[end - 1];
        if (!vowel(c) && c != 'w' && c != 'x' && c != 'Y' && vowel(w[end - 2]) && !vowel(w[end - 3])) return true;
    }
    if (end == 2 && vowel(w[0]) && !vowel(w[1])) return true;  // GCOVR_EXCL_BR_LINE: where called, two letters are a vowel and a consonant (R1 starts there)
    return end >= 4 && std::string_view(w).substr(end - 4, 4) == "past";
}

/// Where a region starts: after the first non-vowel that follows a vowel, from `from`
/// (w.size() if there is none).
std::size_t region(const std::string& w, std::size_t from) {
    std::size_t i = from;
    while (i < w.size() && !vowel(w[i])) ++i;
    while (i < w.size() && vowel(w[i])) ++i;
    return i < w.size() ? i + 1 : w.size();
}

using Rule = std::pair<std::string_view, std::string_view>;  // suffix, replacement

/// The longest suffix of `w` in `rules` (listed longest first), or nullptr.
template <std::size_t N>
const Rule* longest(const std::string& w, const std::array<Rule, N>& rules) {
    for (const Rule& r : rules)
        if (ends(w, r.first)) return &r;
    return nullptr;
}

void replace_end(std::string& w, std::size_t n, std::string_view by) { w.replace(w.size() - n, n, by); }

constexpr std::array<Rule, 25> kStep2{{{"ization", "ize"}, {"ational", "ate"}, {"fulness", "ful"}, {"ousness", "ous"}, {"iveness", "ive"},
                                       {"tional", "tion"}, {"biliti", "ble"}, {"lessli", "less"}, {"entli", "ent"}, {"ation", "ate"},
                                       {"alism", "al"}, {"aliti", "al"}, {"ousli", "ous"}, {"iviti", "ive"}, {"fulli", "ful"},
                                       {"ogist", "og"}, {"enci", "ence"}, {"anci", "ance"}, {"abli", "able"}, {"izer", "ize"},
                                       {"ator", "ate"}, {"alli", "al"}, {"bli", "ble"}, {"ogi", "og"}, {"li", ""}}};
constexpr std::array<Rule, 9> kStep3{{{"ational", "ate"}, {"tional", "tion"}, {"alize", "al"}, {"icate", "ic"}, {"iciti", "ic"},
                                      {"ative", ""}, {"ical", "ic"}, {"ness", ""}, {"ful", ""}}};
constexpr std::array<Rule, 18> kStep4{{{"ement", ""}, {"ance", ""}, {"ence", ""}, {"able", ""}, {"ible", ""}, {"ment", ""},
                                       {"ant", ""}, {"ent", ""}, {"ism", ""}, {"ate", ""}, {"iti", ""}, {"ous", ""}, {"ive", ""},
                                       {"ize", ""}, {"ion", ""}, {"al", ""}, {"er", ""}, {"ic", ""}}};
constexpr std::array<Rule, 15> kExceptions{{{"skis", "ski"}, {"skies", "sky"}, {"idly", "idl"}, {"gently", "gentl"}, {"ugly", "ugli"},
                                            {"early", "earli"}, {"only", "onli"}, {"singly", "singl"}, {"sky", "sky"}, {"news", "news"},
                                            {"howe", "howe"}, {"atlas", "atlas"}, {"cosmos", "cosmos"}, {"bias", "bias"}, {"andes", "andes"}}};

/// Step 1b's rule for -ed, -edly, -ing, -ingly (`n` letters): drop it after a vowel,
/// then restore an "e", undouble, or add "e" after a short syllable.
void step1b_suffix(std::string& w, std::size_t n, std::size_t p1) {
    if (n > w.size()) return;  // GCOVR_EXCL_BR_LINE: callers pass a suffix the word ends with (said here for the compiler)
    if (!has_vowel(w, w.size() - n)) return;
    w.resize(w.size() - n);
    if (ends(w, "at") || ends(w, "bl") || ends(w, "iz")) {
        w += 'e';
    } else if (w.size() >= 2 && w.back() == w[w.size() - 2] && std::string_view("bdfgmnprt").find(w.back()) != std::string_view::npos) {
        const bool kept = w.size() == 3 && (w[0] == 'a' || w[0] == 'e' || w[0] == 'o');  // add, egg, off
        if (!kept) w.pop_back();
    } else if (w.size() == p1 && short_syllable(w, w.size())) {
        w += 'e';
    }
}

}  // namespace

std::string stem_english(std::string_view word) {
    for (const Rule& e : kExceptions)
        if (word == e.first) return std::string(e.second);
    if (word.size() < 3) return std::string(word);
    std::string w(word);
    // Prelude: no leading apostrophe; y as a consonant (Y) at the start and after a vowel.
    if (w[0] == '\'') w.erase(0, 1);
    for (std::size_t i = 0; i < w.size(); ++i)
        if (w[i] == 'y' && (i == 0 || vowel(w[i - 1]))) w[i] = 'Y';
    // Regions: special prefixes, else after the first vowel-consonant pair; R2 likewise after R1.
    std::size_t p1 = w.size();
    bool prefixed = false;
    for (std::string_view prefix : {"gener", "commun", "arsen", "past", "univers", "later", "emerg", "organ", "inter"})
        if (std::string_view(w).substr(0, prefix.size()) == prefix) {
            p1 = prefix.size();
            prefixed = true;
            break;
        }
    if (!prefixed) p1 = region(w, 0);
    const std::size_t p2 = p1 < w.size() ? region(w, p1) : w.size();
    // Step 1a: apostrophes, then plurals.
    for (std::string_view a : {"'s'", "'s", "'"})
        if (ends(w, a)) {
            w.resize(w.size() - a.size());
            break;
        }
    if (ends(w, "sses")) {
        replace_end(w, 4, "ss");
    } else if (ends(w, "ied") || ends(w, "ies")) {
        replace_end(w, 3, w.size() > 4 ? "i" : "ie");  // ties -> tie, cries -> cri
    } else if (ends(w, "us") || ends(w, "ss")) {
    } else if (ends(w, "s") && w.size() >= 2 && has_vowel(w, w.size() - 2)) {
        w.pop_back();  // a vowel before the letter before the s: gaps -> gap, gas stays
    }
    // Step 1b.
    if (ends(w, "eedly") || ends(w, "eed")) {
        const std::size_t n = ends(w, "eedly") ? 5 : 3, start = w.size() - n;
        const std::string_view before = std::string_view(w).substr(0, start);
        if (start >= p1 && before != "proc" && before != "exc" && before != "succ") replace_end(w, n, "ee");
    } else if (ends(w, "ingly") || ends(w, "edly")) {
        step1b_suffix(w, ends(w, "ingly") ? 5 : 4, p1);
    } else if (ends(w, "ing")) {
        const std::string_view before = std::string_view(w).substr(0, w.size() - 3);
        if (before.size() == 2 && before[1] == 'y' && !vowel(before[0])) {
            w = std::string(1, before[0]) + "ie";  // dying -> die
        } else if (before != "inn" && before != "out" && before != "cann" && before != "herr" && before != "earr" && before != "even") {
            step1b_suffix(w, 3, p1);
        }
    } else if (ends(w, "ed")) {
        step1b_suffix(w, 2, p1);
    }
    // Step 1c: a final y after a consonant (not the first letter) becomes i.
    if (w.size() > 2 && (w.back() == 'y' || w.back() == 'Y') && !vowel(w[w.size() - 2])) w.back() = 'i';
    // Steps 2 to 4 remove suffixes only inside R1 or R2, which start at index 2 or later
    // (after a vowel and a consonant, a prefix, or the end of a word of three letters or
    // more): the letter before a suffix always exists.
    // Steps 2 and 3, in R1.
    if (const Rule* r = longest(w, kStep2); r && w.size() - r->first.size() >= p1) {
        const std::size_t start = w.size() - r->first.size();
        if (r->first == "ogi") {
            if (w[start - 1] == 'l') replace_end(w, 3, "og");
        } else if (r->first == "li") {
            if (std::string_view("cdeghkmnrt").find(w[start - 1]) != std::string_view::npos) w.resize(start);
        } else {
            replace_end(w, r->first.size(), r->second);
        }
    }
    if (const Rule* r = longest(w, kStep3); r && w.size() - r->first.size() >= p1)
        if (r->first != "ative" || w.size() - 5 >= p2) replace_end(w, r->first.size(), r->second);
    // Step 4, in R2: -ion only after s or t.
    if (const Rule* r = longest(w, kStep4); r && w.size() - r->first.size() >= p2) {
        const std::size_t start = w.size() - r->first.size();
        if (r->first != "ion" || w[start - 1] == 's' || w[start - 1] == 't') w.resize(start);
    }
    // Step 5: a final e in R2, or in R1 unless after a short syllable; ll in R2.
    if (ends(w, "e")) {
        const std::size_t start = w.size() - 1;
        if (start >= p2 || (start >= p1 && !short_syllable(w, start))) w.pop_back();
    } else if (ends(w, "ll") && w.size() - 1 >= p2) {
        w.pop_back();
    }
    for (char& c : w)
        if (c == 'Y') c = 'y';
    return w;
}

namespace {

constexpr std::array<std::string_view, 174> kStopWords{{
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"}};

/// Decodes one UTF-8 character at `i`: its code point and length, or 0xFFFFFFFF and 1
/// for an invalid sequence (an overlong form, a surrogate, beyond U+10FFFF, a bad byte).
std::pair<char32_t, std::size_t> decode(std::string_view s, std::size_t i) {
    const auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[i + k]); };
    const unsigned char c = b(0);
    if (c < 0x80) return {c, 1};
    const std::size_t n = c >= 0xF0 && c < 0xF5 ? 4 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xC2 && c < 0xE0 ? 2 : 0;
    if (n == 0 || i + n > s.size()) return {0xFFFFFFFF, 1};
    char32_t cp = c & (0x7F >> n);
    for (std::size_t k = 1; k < n; ++k) {
        if ((b(k) & 0xC0) != 0x80) return {0xFFFFFFFF, 1};
        cp = (cp << 6) | (b(k) & 0x3F);
    }
    const char32_t least[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < least[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return {0xFFFFFFFF, 1};
    return {cp, n};
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// Letters and digits (in a token); everything else separates tokens.
bool word_char(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    if (c == 0xFFFFFFFF || c < 0xA0) return false;                                       // invalid; C1 controls
    if (c <= 0xBF) return c == 0xAA || c == 0xB5 || c == 0xBA;                            // Latin-1 punctuation and symbols
    if (c == 0xD7 || c == 0xF7) return false;                                            // multiplication, division
    if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3000 && c <= 0x303F)) return false;  // spaces, punctuation, symbols
    if ((c >= 0xFE30 && c <= 0xFE4F) || c == 0xFEFF || (c >= 0xFF00 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) ||
        (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65))
        return false;                                                                    // CJK and fullwidth punctuation
    return !(c >= 0x1F000 && c <= 0x1FAFF);                                              // emoji and pictographs
}

char32_t lower(char32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    if (c >= 0xC0 && c <= 0xDE) return c + 0x20;  // Latin-1 (U+00D7, inside the range, is a separator: never here)
    if (c == 0x130) return 'i';  // İ, the one irregular letter of Latin Extended-A: before its rule (which would give ı)
    if ((c >= 0x100 && c <= 0x137) || (c >= 0x14A && c <= 0x177)) return c | 1;          // Latin Extended-A: even upper, odd lower
    if ((c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E)) return (c & 1) ? c + 1 : c;  // odd upper, even lower
    if (c == 0x178) return 0xFF;                                                         // Ÿ
    if (c >= 0x391 && c <= 0x3AB && c != 0x3A2) return c + 0x20;                         // Greek
    if (c == 0x386) return 0x3AC;
    if (c >= 0x388 && c <= 0x38A) return c + 0x25;
    if (c == 0x38C) return 0x3CC;
    if (c == 0x38E || c == 0x38F) return c + 0x3F;
    if (c >= 0x410 && c <= 0x42F) return c + 0x20;                                       // Cyrillic
    if (c >= 0x400 && c <= 0x40F) return c + 0x50;
    return c;
}

bool apostrophe(char32_t c) { return c == '\'' || c == 0x2019; }

}  // namespace

std::vector<std::string> tokenize_unicode(std::string_view text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (std::size_t i = 0; i < text.size();) {
        const auto [cp, n] = decode(text, i);
        i += n;
        if (word_char(cp)) {
            append_utf8(cur, lower(cp));
        } else if (apostrophe(cp) && !cur.empty() && i < text.size() && word_char(decode(text, i).first)) {
            cur += '\'';  // inside a word: don't, john's
        } else if (!cur.empty()) {
            tokens.push_back(std::move(cur));
            cur.clear();
        }
    }
    if (!cur.empty()) tokens.push_back(std::move(cur));
    return tokens;
}

bool is_english_stop_word(std::string_view token) { return std::binary_search(kStopWords.begin(), kStopWords.end(), token); }

}  // namespace vecdb